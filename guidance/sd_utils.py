from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        control_mode="normal",
        model_key="runwayml/stable-diffusion-v1-5",
    ):
        super().__init__()

        self.device = device
        self.control_mode = control_mode
        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.precision_t
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        # controlnet
        if self.control_mode is not None:
            # NOTE: controlnet 1.1 is much better...
            if self.control_mode == "normal":
                self.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_normalbae",
                    torch_dtype=self.precision_t,
                ).to(self.device)
            elif self.control_mode == "ip2p":
                self.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11e_sd15_ip2p",
                    torch_dtype=self.precision_t,
                ).to(self.device)
            else:
                raise NotImplementedError
            
            self.controlnet_conditioning_scale = 1.0

        # self.scheduler = DDIMScheduler.from_pretrained(
        #     model_key, subfolder="scheduler", torch_dtype=self.precision_t
        # )

        self.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

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
        num_inference_steps=20,
        guidance_scale=7.5,
        control_image=None,
        latents=None,
    ):

        text_embeddings = text_embeddings.to(self.precision_t)

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, 4, height // 8, width // 8,), dtype=self.precision_t, device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # controlnet
            if self.control_mode is not None and control_image is not None:
                control_image_input = torch.cat([control_image] * 2).to(self.precision_t)
                down_block_res_samples, mid_block_res_sample = self.controlnet(latent_model_input, t, encoder_hidden_states=text_embeddings, controlnet_cond=control_image_input, return_dict=False)
                down_block_res_samples = [down_block_res_sample * self.controlnet_conditioning_scale for down_block_res_sample in down_block_res_samples]
                mid_block_res_sample *= self.controlnet_conditioning_scale

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings, 
                    down_block_additional_residuals=down_block_res_samples, 
                    mid_block_additional_residual=mid_block_res_sample
                ).sample
            else:
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings,
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        return imgs