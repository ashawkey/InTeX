import numpy as np
import gradio as gr
import argparse

from gui import GUI

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=str, default=None)
parser.add_argument("--prompt", type=str, default='')
parser.add_argument("--posi_prompt", type=str, default="high quality")
parser.add_argument("--nega_prompt", type=str, default="worst quality, low quality")
parser.add_argument("--control_mode", action='append', default=['normal', 'inpaint'])
# parser.add_argument("--control_mode", default=None)
parser.add_argument("--outdir", type=str, default="logs")
parser.add_argument("--save_path", type=str, default="out")
# parser.add_argument("--model_key", type=str, default="stablediffusionapi/anything-v5")
# parser.add_argument("--model_key", type=str, default="xyn-ai/anything-v4.0")
parser.add_argument("--model_key", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--wogui", action='store_true')
parser.add_argument("--text_dir", action='store_true')
parser.add_argument("--H", type=int, default=800)
parser.add_argument("--W", type=int, default=800)
parser.add_argument("--radius", type=float, default=2)
parser.add_argument("--fovy", type=float, default=60)

opt = parser.parse_args()
opt.wogui = True
opt.text_dir = True
opt.save_path = 'out.glb' # use glb to correctly show texture in gr.Model3D

core = GUI(opt)
core.prepare_guidance()

def process(prompt, mesh_path):
    core.prompt = opt.posi_prompt + ', ' + prompt
    core.renderer.load_mesh(mesh_path)
    core.generate()
    out_path = core.save_model()
    # return out_path
    tex = core.renderer.mesh.albedo.detach().cpu().numpy()
    tex = (tex * 255).astype(np.uint8)
    return out_path, tex

block = gr.Blocks().queue()
with block:
    gr.Markdown("## Tetere: text-to-texture")

    with gr.Row():
        with gr.Column():
            input_prompt = gr.Textbox(label="prompt")
            model = gr.Model3D(label="3D model")
            button_generate = gr.Button("Generate")
        
        with gr.Column():
            output_image = gr.Image(label="texture")
    
        button_generate.click(process, inputs=[input_prompt, model], outputs=[model, output_image])
    
block.launch(server_name="0.0.0.0")