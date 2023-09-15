import os
import math
import cv2
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from mesh import Mesh, safe_normalize

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)

class Renderer(nn.Module):
    def __init__(self, device, opt):
        
        super().__init__()

        self.device = device
        self.opt = opt

        self.mesh = None

        if opt.bg_image is not None and os.path.exists(opt.bg_image):
            # load an image as the background
            bg_image = cv2.imread(opt.bg_image)
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            bg_image = torch.from_numpy(bg_image.astype(np.float32) / 255).to(self.device)
            self.bg = F.interpolate(bg_image.permute(2, 0, 1).unsqueeze(0), (opt.render_resolution, opt.render_resolution), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).contiguous()
        else:
            self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        self.bg_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        if not self.opt.gui or os.name == 'nt':
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

    @torch.no_grad()
    def load_mesh(self, path, front_dir):
        self.mesh = Mesh.load(path, front_dir=front_dir, device=self.device)

    @torch.no_grad()
    def export_mesh(self, path):
        self.mesh.write(path)
        
    def render(self, pose, proj, h, w):

        results = {}

        # get v
        v = self.mesh.v

        # get v_clip and render rgb
        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T
        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))

        # actually disparity (1 / depth), to align with controlnet
        disp = -1 / (v_cam[..., [2]] + 1e-20)
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-20) # pre-normalize
        depth, _ = dr.interpolate(disp, rast, self.mesh.f) # [1, H, W, 1]
        depth = depth.clamp(0, 1).squeeze(0) # [H, W, 1]

        alpha = (rast[..., 3:] > 0).float()

        # rgb texture
        texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
        albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear') # [1, H, W, 3]
        
        # get vn and render normal
        vn = self.mesh.vn
        
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]

        # rot normal z axis is exactly viewdir-normal cosine
        viewcos = rot_normal[..., [2]]

        # antialias
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

        # replace background
        albedo = alpha * albedo + (1 - alpha) * self.bg
        normal = alpha * normal + (1 - alpha) * self.bg_normal
        rot_normal = alpha * rot_normal + (1 - alpha) * self.bg_normal

        # extra texture (hard coded)
        if hasattr(self.mesh, 'cnt'):
            cnt = dr.texture(self.mesh.cnt.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            cnt = dr.antialias(cnt, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            cnt = alpha * cnt + (1 - alpha) * 1 # 1 means no-inpaint in background
            results['cnt'] = cnt
        
        if hasattr(self.mesh, 'viewcos_cache'):
            viewcos_cache = dr.texture(self.mesh.viewcos_cache.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            viewcos_cache = dr.antialias(viewcos_cache, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            results['viewcos_cache'] = viewcos_cache

        if hasattr(self.mesh, 'ori_albedo'):
            ori_albedo = dr.texture(self.mesh.ori_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            ori_albedo = dr.antialias(ori_albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            ori_albedo = alpha * ori_albedo + (1 - alpha) * self.bg
            results['ori_image'] = ori_albedo
        
        # all shaped as [H, W, C]
        results['image'] = albedo
        results['alpha'] = alpha
        results['depth'] = depth
        results['normal'] = normal # in [-1, 1]
        results['rot_normal'] = rot_normal # in [-1, 1]
        results['viewcos'] = viewcos
        results['uvs'] = texc.squeeze(0)

        return results