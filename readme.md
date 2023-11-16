# InteX

This repository contains the official implementation for [InteX: Interactive Text-to-Texture Synthesis via Unified Depth-aware Inpainting](TODO).

![teaser](assets/teaser.jpg)

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
python main.py --config configs/revani.yaml mesh=data/dragon.glb prompt="a red pet dragon with fire patterns" save_path=dragon_fire.glb text_dir=True

### visualize generated mesh (requires to install kiui)
kire ./logs/dragon_fire.glb

### web gui (gradio)
python app.py
```

Please check `./scripts` for more examples.

Tips:
* support loading/saving `obj, ply, glb` formated meshes.


### Others

```bash
### Automatically download [objaverse](https://objaverse.allenai.org/explore) model by uid:
# auto download the model, and run tex gen.
python main.py --config configs/base.yaml mesh='u3WYrMucGzUOhnNukx2EfyQqevA' prompt="a photo of game controller" save_path=controller.obj

# just visualize the downloaded model with original texture
python main.py --config configs/base.yaml mesh='4c865c5df17741f7be87392d37fa31eb' prompt='xxx' gui=True

### interactive tools
# visualize intermediate results
python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="a photo of napoleon" save_path=napoleon.obj text_dir=True vis=True

# use a local GUI
python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="a photo of napoleon" save_path=napoleon.obj text_dir=True gui=True
```
