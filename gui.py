import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import trimesh
import rembg

from cam_utils import orbit_camera, OrbitCamera
from mesh_renderer import Renderer

from mesh import Mesh

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.wogui = opt.wogui # disable gui and run in cmd
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
        self.renderer = Renderer(opt)

        # input text
        self.prompt = self.opt.input
        self.negative_prompt = ""

        if not self.wogui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if not self.wogui:
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
        
        from guidance.sd_utils import StableDiffusion
        self.guidance = StableDiffusion(self.device, t_range=[0.02, 0.5])

        nega = self.guidance.get_text_embeds([self.negative_prompt])
        posi = self.guidance.get_text_embeds([self.prompt])
        self.guidance_embeds = torch.cat([nega] * self.opt.batch_size + [posi] * self.opt.batch_size, dim=0)
            
    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / 500)

            loss = 0

            ### novel view (manual batch)
            images = []
            vers, hors, radii = [], [], []
            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(-30, 10)
                hor = np.random.randint(-180, 180)
                radius = self.opt.radius #+ np.random.random() # [r, r + 1]

                vers.append(ver)
                hors.append(hor)
                radii.append(radius - self.opt.radius)

                pose = orbit_camera(ver, hor, radius)
                mvp = self.cam.perspective @ np.linalg.inv(pose)
                mvp = torch.from_numpy(mvp.astype(np.float32)).to(self.device)

                # random resolution
                ssaa = 2 # min(1.0, max(0.125, np.random.random()))
                out = self.renderer.render(mvp, 256, 256, ssaa=ssaa)

                rand = np.random.random()
                if True: #rand < 0.5:
                    image = out["image"] # [H, W, 3] in [0, 1]
                else:
                    # random normal-consistency loss from TADA!
                    mix_ratio = min(0.9, max(0.1, np.random.random()))
                    image = mix_ratio * out["image"].detach() + (1 - mix_ratio) * out["normal"]
                
                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                images.append(image)
            
            images = torch.cat(images, dim=0)

            # import kiui
            # kiui.lo(hor, ver)
            # kiui.vis.plot_image(image)

            # guidance loss
            if self.opt.guidance_model == 'zero123':

                refined_images = self.guidance.refine(self.guidance_embeds, images, vers, hors, radii)
                loss = loss + F.mse_loss(images, refined_images)
                # import kiui
                # # print(vers, hors)
                # kiui.vis.plot_image(images, refined_images)

                # loss = loss + self.guidance.train_step(self.guidance_embeds, images, vers, hors, radii)
            else:
                loss = loss + self.guidance.train_step(self.guidance_embeds, images)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if not self.wogui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

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

            mvp = torch.from_numpy(self.cam.mvp.astype(np.float32)).to(self.device)

            out = self.renderer.render(mvp, self.H, self.W)

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

        if not self.wogui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!


    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.obj')
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")

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

                if self.opt.guidance_model in ["SD", "IF"]:
                    dpg.add_input_text(
                        label="negative",
                        default_value=self.negative_prompt,
                        callback=callback_setattr,
                        user_data="negative_prompt",
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=self.save_model,
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
        assert not self.wogui
        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def run(self):
        self.save_model()
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--wogui", action='store_true')
    parser.add_argument("--lock_geo", action='store_true')
    parser.add_argument("--H", type=int, default=800)
    parser.add_argument("--W", type=int, default=800)
    parser.add_argument("--iters", type=int, default=100)
    # parser.add_argument("--ssaa", type=float, default=1)
    parser.add_argument("--radius", type=float, default=5)
    parser.add_argument("--fovy", type=float, default=20)
    parser.add_argument("--outdir", type=str, default="logs")
    parser.add_argument("--save_path", type=str, default="out2")
    parser.add_argument("--guidance_model", type=str, default="SD", choices=["none", "zero123", "IF", "SD", "clip"])
    parser.add_argument("--batch_size", type=int, default=1)

    opt = parser.parse_args()

    gui = GUI(opt)

    if opt.wogui:
        gui.run()
    else:
        gui.render()
