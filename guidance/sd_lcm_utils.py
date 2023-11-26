import os
from transformers import logging
from diffusers import (
    DDIMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F


# ref: https://github.com/huggingface/diffusers/blob/v0.20.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L58C1-L69C21
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        control_mode=["normal",],
        model_key="runwayml/stable-diffusion-v1-5",
        lora_keys=[],
    ):
        super().__init__()

        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        self.control_mode = control_mode

        # Create model
        if os.path.exists(model_key):
            # treat as local ckpt
            pipe = StableDiffusionPipeline.from_single_file(model_key, torch_dtype=self.dtype)
        else:
            # huggingface ckpt
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype)
        
        for lora_key in lora_keys:
            # assume local folder/xxx.safetensors
            assert os.path.exists(lora_key)
            folder = os.path.dirname(lora_key)
            name = os.path.basename(lora_key)
            pipe.load_lora_weights(folder, weight_name=name)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        # controlnet
        if self.control_mode is not None:
            self.controlnet = {}
            self.controlnet_conditioning_scale = {}
            
            if "normal" in self.control_mode:
                self.controlnet['normal'] = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae",torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['normal'] = 1.0
            if "depth" in self.control_mode:
                self.controlnet['depth'] = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['depth'] = 1.0
            if "ip2p" in self.control_mode:
                self.controlnet['ip2p'] = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_ip2p",torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['ip2p'] = 1.0
            if "inpaint" in self.control_mode:
                self.controlnet['inpaint'] = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['inpaint'] = 1.0
            if "depth_inpaint" in self.control_mode:
                self.controlnet['depth_inpaint'] = ControlNetModel.from_pretrained("ashawkey/control_v11e_sd15_depth_aware_inpaint",torch_dtype=self.dtype).to(self.device)
                # self.controlnet['depth_inpaint'] = ControlNetModel.from_pretrained("ashawkey/controlnet_depth_aware_inpaint_v11", torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['depth_inpaint'] = 1.0
            
        # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.dtype)
        # self.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        self.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        del pipe

       
    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=2.0,
        guidance_rescale=0,
        control_images=None,
        latents=None,
        strength=0,
        refine_strength=0.8,
    ):

        text_embeddings = text_embeddings.to(self.dtype)
        for k in control_images:
            control_images[k] = control_images[k].to(self.dtype)

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, 4, height // 8, width // 8,), dtype=self.dtype, device=self.device)
        
        if strength != 0:
            full_num_inference_steps = int(num_inference_steps / (1 - strength))
            self.scheduler.set_timesteps(full_num_inference_steps)
            init_step = full_num_inference_steps - num_inference_steps
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        else:
            self.scheduler.set_timesteps(num_inference_steps)
            init_step = 0

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            
            # inpaint mask blend
            if 'latents_mask' in control_images:
                if i < num_inference_steps * refine_strength:
                    # fix keep + refine at early steps
                    mask_keep = 1 - control_images['latents_mask']
                else:
                    # only fix keep at later steps
                    mask_keep = control_images['latents_mask_keep']

                latents_original = control_images['latents_original']
                latents_original_noisy = self.scheduler.add_noise(latents_original, torch.randn_like(latents_original), t)
                latents = latents * (1 - mask_keep) + latents_original_noisy * mask_keep

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # controlnet
            if self.control_mode is not None and control_images is not None:

                noise_pred = 0

                for mode, controlnet in self.controlnet.items():
                    # may omit control mode if input is not provided
                    if mode not in control_images: continue
                    
                    control_image = control_images[mode]
                    weight = 1 / len(self.controlnet)

                    control_image_input = torch.cat([control_image] * 2)
                    down_samples, mid_sample = controlnet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings, 
                        controlnet_cond=control_image_input, 
                        conditioning_scale=self.controlnet_conditioning_scale[mode],
                        return_dict=False
                    )

                    # predict the noise residual
                    noise_pred_cur = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings, 
                        down_block_additional_residuals=down_samples, 
                        mid_block_additional_residual=mid_sample
                    ).sample

                    # merge after unet
                    noise_pred = noise_pred + weight * noise_pred_cur
                
            else:
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings,
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            if guidance_rescale > 0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        return imgs