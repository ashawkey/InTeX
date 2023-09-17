# Tetere

Tetere is a TExt-to-TExtuRE tool.

### Features
* Text-to-Texture in one minute.
* Support ControlNet v1.1 inpaint, ip2p, depth, and normal control to generate consistent texture.
* Support custom checkpoints (need to be converted to diffuser format)

### Install
```bash
pip install -r requirements.txt
```

### Usage
```bash
# run
python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="a photo of napoleon" save_path=napoleon.obj text_dir=True

# visualize intermediate results
python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="a photo of napoleon" save_path=napoleon.obj text_dir=True vis=True

# interactive inpaint (using cv2 GUI, press space=accept, other=reject)
python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="a photo of napoleon" save_path=napoleon.obj text_dir=True interactive=True

# [experimental] gradio web gui (only allow obj/glb/gltf)
python app.py
```

Automatically download [objaverse](https://objaverse.allenai.org/explore) model by uid:
```bash
# auto download the model, and run tex gen.
python main.py --config configs/base.yaml mesh='u3WYrMucGzUOhnNukx2EfyQqevA' prompt="a photo of game controller" save_path=controller.obj

# just visualize the downloaded model with original texture
python main.py --config mesh='u3WYrMucGzUOhnNukx2EfyQqevA' prompt='xxx' gui=True
```