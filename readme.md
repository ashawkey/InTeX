# Tetere

Tetere is a TExt-to-TExtuRE tool.

https://github.com/ashawkey/tetere_private/assets/25863658/5774f660-1839-488b-9967-40996490aad4

### Features
* Text-to-Texture in one minute.
* Support ControlNet v1.1 inpaint, ip2p, depth, and normal control to generate consistent texture.
* Support custom checkpoints (need to be converted to diffuser format)

### Install
```bash
pip install -r requirements.txt

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/
```

### Usage
```bash
### generate texture for a mesh based on a prompt (command line), output will be saved to ./logs
# please check ./configs/revani.yaml for each parameter's meaning
python main.py --config configs/revani.yaml mesh=data/dragon.obj prompt="a red pet dragon with fire patterns" save_path=dragon_fire.obj text_dir=True 

### visualize generated mesh
python -m kiui.render ./logs/dragon_fire.obj

### web gui (gradio)
python app.py
```

Please check `./scripts` for more examples.

Tips:
* support loading `obj, ply, glb` formated meshes.
* output will be in `obj` format with seperately saved texture images.


### Others

```bash
### Automatically download [objaverse](https://objaverse.allenai.org/explore) model by uid:
# auto download the model, and run tex gen.
python main.py --config configs/base.yaml mesh='u3WYrMucGzUOhnNukx2EfyQqevA' prompt="a photo of game controller" save_path=controller.obj

# just visualize the downloaded model with original texture
python main.py --config mesh='u3WYrMucGzUOhnNukx2EfyQqevA' prompt='xxx' gui=True

### interactive tools
# visualize intermediate results
python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="a photo of napoleon" save_path=napoleon.obj text_dir=True vis=True
```
