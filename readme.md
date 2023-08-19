# Tetere

Tetere is a TExt-to-TExtuRE tool.

### Features
* Text-to-Texture in one minute.
* Use ControlNet v1.1 inpaint and normal model to generate 3D consistent texture.
* Support custom checkpoints (need to be converted to diffuser format)

### Install
```bash
pip install -r requirements.txt
```

### Usage
```bash
# gui
python gui.py --mesh data/dragon.obj --prompt "a pet dragon with rainbow patterns" --save_path dragon_rainbow --text_dir

# cmd
python gui.py --mesh data/dragon.obj --prompt "a pet dragon with rainbow patterns" --save_path dragon_rainbow --text_dir --wogui
```