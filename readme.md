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

# gradio web gui (only allow obj/glb/gltf)
python app.py
```

Automatically download [objaverse](https://objaverse.allenai.org/explore) model by uid:
```bash
python main.py --config configs/base.yaml mesh=u3WYrMucGzUOhnNukx2EfyQqevA prompt="a photo of game controller" save_path=controller.obj
```