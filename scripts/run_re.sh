python main.py --config configs/base.yaml mesh=data_objaverse/guitar.glb prompt="a cool guitar" save_path=guitar.obj text_dir=False front_dir=+x1 retex=True
python main.py --config configs/base.yaml mesh=data_objaverse/leather_bag.glb prompt="a leather bag" save_path=bag.obj text_dir=False front_dir=-y
python main.py --config configs/revani.yaml mesh=data_objaverse/penguin.glb camera_path=side prompt="a cute white and black penguin wearing a backpack" save_path=penguin.obj text_dir=True
python main.py --config configs/revani.yaml mesh='8ab62952a2704eeb9a8e36c31b95d9b1' prompt="a girl with brown hair" save_path=girl.obj text_dir=True
python main.py --config configs/base.yaml mesh='aa8a074fae9a49f0a1b26b87ac826802' prompt="an old chinese temple" save_path=temple.obj retex=True text_dir=True