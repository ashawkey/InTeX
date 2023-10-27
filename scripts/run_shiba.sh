python main.py --config configs/base.yaml mesh=data_objaverse/shiba.glb prompt="a shiba dog" save_path=shiba.obj text_dir=True front_dir=+z 
python main.py --config configs/revani.yaml mesh=data_objaverse/shiba.glb prompt="a shiba dog" save_path=shiba_revani.obj text_dir=True front_dir=+z 
python main.py --config configs/anything.yaml mesh=data_objaverse/shiba.glb prompt="a shiba dog" save_path=shiba_anything.obj text_dir=True front_dir=+z 

python -m kiui.render logs/shiba.obj --wogui --save videos/shiba --radius 2.5
python -m kiui.render logs/shiba_revani.obj --wogui --save videos/shiba_revani --radius 2.5
python -m kiui.render logs/shiba_anything.obj --wogui --save videos/shiba_anything --radius 2.5
