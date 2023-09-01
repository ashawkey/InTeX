import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from cam_utils import orbit_camera, OrbitCamera
from mesh_renderer import Renderer

from grid_put import mipmap_linear_grid_put_2d, nearest_grid_put_2d

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")

        self.guidance = None
        self.guidance_embeds = None

        # renderer
        self.renderer = Renderer(self.device, opt)

        # input mesh
        if self.opt.mesh is not None:
            self.renderer.load_mesh(self.opt.mesh, self.opt.front_dir)

        # input text
        self.prompt = self.opt.posi_prompt + ', ' + self.opt.prompt
        self.negative_prompt = self.opt.nega_prompt

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed
    
    def prepare_guidance(self):
        
        print(f'[INFO] loading guidance model...')

        from guidance.sd_utils import StableDiffusion
        self.guidance = StableDiffusion(self.device, control_mode=self.opt.control_mode, model_key=self.opt.model_key)

        nega = self.guidance.get_text_embeds([self.negative_prompt])

        if not self.opt.text_dir:
            posi = self.guidance.get_text_embeds([self.prompt])
            self.guidance_embeds = torch.cat([nega, posi], dim=0)
        else:
            self.guidance_embeds = {}
            for d in ['front', 'side', 'back', 'overhead', 'bottom']:
                posi = self.guidance.get_text_embeds([self.prompt + f', {d} view'])
                self.guidance_embeds[d] = torch.cat([nega, posi], dim=0)

        
        print(f'[INFO] loaded guidance model!')

    @torch.no_grad()
    def generate(self, texture_size=1024, render_resolution=512):

        print(f'[INFO] start generation...')

        if self.guidance is None:
            self.prepare_guidance()
        
        h = w = int(texture_size)
        H = W = int(render_resolution)

        albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
        cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)
        viewcos_cache = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

        # keep original texture if using ip2p
        if 'ip2p' in self.opt.control_mode:
            self.renderer.mesh.ori_albedo = self.renderer.mesh.albedo.clone()
        # init empty texture, and also patch texture-cnt to mesh for rendering inpaint mask
        self.renderer.mesh.albedo = albedo
        self.renderer.mesh.cnt = cnt 
        self.renderer.mesh.viewcos_cache = viewcos_cache

        # vers = [0,]
        # hors = [0,]

        if not self.opt.text_dir:
            vers = [-30] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0]
        else:
            # spiral-like camera path...
            vers = [0, -45, 0,    0, -89.9, 89.9,  0,   0,   0,    0,   0]
            hors = [0, 0,   45, -45,     0,    0, 90, -90, 135, -135, 180]

        # better to generate a top-back-view earlier
        # vers = [0, -45, -45,  0,   0, -89.9,  0,   0, 89.9,   0,    0]
        # hors = [0, 180,   0, 45, -45,     0, 90, -90,    0, 135, -135]

        start_t = time.time()

        first_iter = True

        for ver, hor in zip(vers, hors):
            # render image
            pose = orbit_camera(ver, hor, self.cam.radius)
            out = self.renderer.render(pose, self.cam.perspective, H, W)

            # valid crop region with fixed aspect ratio
            valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
            min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
            min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()
            
            size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
            h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
            w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

            min_h = int(h_start)
            min_w = int(w_start)
            max_h = int(min_h + size)
            max_w = int(min_w + size)

            def _zoom(x, mode='bilinear'):
                return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], (512, 512), mode=mode)

            image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]

            # trimap: generate
            mask_generate = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous()) < 0.1 # [1, 1, H, W]
            mask_generate = gaussian_blur(mask_generate.float(), kernel_size=15, sigma=5) # [1, 1, H, W]
            mask_generate[mask_generate > 0.5] = 1 # do not mix any inpaint region
            
            # trimap: refine (contains generate, opposite of keep)
            viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
            viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]

            mask_refine = (viewcos_old < viewcos).float()

            # import kiui
            # kiui.vis.plot_matrix(viewcos_old, viewcos, mask_refine)

            if not (mask_generate > 0.5).any():
                continue

            control_images = {}

            # construct normal control
            if 'normal' in self.opt.control_mode:
                rot_normal = out['rot_normal'] # [H, W, 3]
                rot_normal[..., 0] *= -1 # align with normalbae: blue = front, red = left, green = top
                control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5) # [1, 3, H, W]
            
            # construct depth control
            if 'depth' in self.opt.control_mode:
                depth = out['depth']
                control_images['depth'] = _zoom(depth.view(1, 1, H, W)).repeat(1, 3, 1, 1) # [1, 3, H, W]
            
            # construct ip2p control
            if 'ip2p' in self.opt.control_mode:
                ori_image = _zoom(out['ori_image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]
                control_images['ip2p'] = ori_image

            # construct inpaint control
            if 'inpaint' in self.opt.control_mode and not first_iter:
                image_generate = image.clone()
                image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region

                image_refine = image.clone()
                image_refine[mask_refine.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region

                control_images['inpaint'] = image_generate
                control_images['inpaint_refine'] = image_refine

                # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
                latents_mask = F.interpolate(mask_generate, size=(H//8, W//8), mode='bilinear') # [1, 1, 64, 64]
                control_images['latents_mask'] = latents_mask
                control_images['latents_original'] = self.guidance.encode_imgs(image.to(self.guidance.dtype)) # [1, 4, 64, 64]
            
            
            if not self.opt.text_dir:
                text_embeds = self.guidance_embeds
            else:
                if ver < -60: d = 'overhead'
                elif ver > 60: d = 'bottom'
                else:
                    if abs(hor) < 60: d = 'front'
                    elif abs(hor) < 120: d = 'side'
                    else: d = 'back'
                text_embeds = self.guidance_embeds[d]

            rgbs = self.guidance(text_embeds, control_images=control_images).float()
            
            # apply mask to make sure non-inpaint region is not changed
            rgbs = image * (1 - mask_generate) + rgbs * mask_generate

            if self.opt.vis:
                import kiui
                if 'depth' in control_images:
                    kiui.vis.plot_image(control_images['depth'])
                if 'normal' in control_images:
                    kiui.vis.plot_image(control_images['normal'])
                if 'ip2p' in control_images:
                    kiui.vis.plot_image(ori_image)
                # kiui.vis.plot_image(mask_generate)
                if 'inpaint' in control_images:
                    kiui.vis.plot_image(control_images['inpaint'].clamp(0, 1))
                    # kiui.vis.plot_image(control_images['inpaint_refine'].clamp(0, 1))
                kiui.vis.plot_image(rgbs)

            # grid put

            # project-texture mask
            proj_mask = (out['alpha'] > 0) & (out['viewcos'] > 0.5)  # [H, W, 1]
            proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
            uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')

            uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
            rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]
            
            print(f'[INFO] processing {ver} - {hor}, {rgbs.shape}')

            cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=256, return_count=True)
            
            # albedo += cur_albedo
            # cnt += cur_cnt
            mask = cnt.squeeze(-1) < 0.1
            albedo[mask] += cur_albedo[mask]
            cnt[mask] += cur_cnt[mask]

            # update mesh texture for rendering
            mask = cnt.squeeze(-1) > 0
            cur_albedo = albedo.clone()
            cur_albedo[mask] /= cnt[mask].repeat(1, 3)
            self.renderer.mesh.albedo = cur_albedo

            # update viewcos cache
            viewcos = viewcos.view(-1, 1)[proj_mask]
            cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)

            # kiui.vis.plot_matrix(cur_viewcos.detach().cpu().numpy())

            self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)

            first_iter = False

        mask = cnt.squeeze(-1) > 0
        albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

        ## dilate texture
        mask = mask.view(h, w)
        albedo = albedo.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        from sklearn.neighbors import NearestNeighbors
        from scipy.ndimage import binary_dilation, binary_erosion

        inpaint_region = binary_dilation(mask, iterations=32)
        inpaint_region[mask] = 0

        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
        _, indices = knn.kneighbors(inpaint_coords)

        albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

        self.renderer.mesh.albedo = torch.from_numpy(albedo).to(self.device)
        # self.renderer.mesh.albedo = albedo

        torch.cuda.synchronize()
        end_t = time.time()
        print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')

        self.need_update = True

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!


    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
        return path

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # prompt stuff
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )
            
                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # generate texture
                with dpg.group(horizontal=True):
                    dpg.add_text("Generate: ")

                    def callback_generate(sender, app_data):
                        self.generate()

                    dpg.add_button(
                        label="gen",
                        tag="_button_gen",
                        callback=callback_generate,
                    )
                    dpg.bind_item_theme("_button_gen", theme_button)
                
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save_model(sender, app_data):
                        self.save_model()

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def run(self):
        self.generate()
        self.save_model()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument("--gpu", default="0", help="GPU ID")

    args, extras = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    print(opt)

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.run()