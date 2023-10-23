import numpy as np
import gradio as gr
from main import GUI
import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configs/revani.yaml', help="path to the yaml config file")
args, extras = parser.parse_known_args()

# override default config from cli
opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

# override some options
opt.prompt = ''
opt.gui = False
opt.save_path = 'gradio_output.glb' # use glb to correctly show texture in gr.Model3D

core = GUI(opt)

def process(prompt, is_text_dir, mesh_path):
    opt.text_dir = is_text_dir
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
            is_text_dir = gr.Checkbox(label="use directional prompt")
            model = gr.Model3D(label="input model")
            button_generate = gr.Button("Generate")
        
        with gr.Column():
            output_model = gr.Model3D(label="output model")
            output_image = gr.Image(label="texture")
    
        button_generate.click(process, inputs=[input_prompt, is_text_dir, model], outputs=[output_model, output_image])
    
block.launch(server_name="0.0.0.0", share=True)