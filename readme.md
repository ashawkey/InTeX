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
# local gui
python gui.py --gpu 0 --config configs/base.yaml mesh=data2/napoleon.obj prompt="a photo of napoleon" save_path=napoleon.obj text_dir=True 

# cmd (turn off gui)
python gui.py --gpu 0 --config configs/base.yaml mesh=data2/napoleon.obj prompt="a photo of napoleon" save_path=napoleon.obj text_dir=True wogui=True

# gradio web gui (only allow obj/glb/gltf)
python app.py
```