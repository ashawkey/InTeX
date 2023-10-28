export CUDA_VISIBLE_DEVICES=7

python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="a cool man with black hair" save_path=man.obj text_dir=True front_dir=+z 
# python main.py --config configs/revani.yaml mesh=data2/napoleon.obj prompt="a cool man with black hair" save_path=man_revani.obj text_dir=True front_dir=+z 
# python main.py --config configs/guofeng.yaml mesh=data2/napoleon.obj prompt="a cool man with black hair" save_path=man_guofeng.obj text_dir=True front_dir=+z 

python -m kiui.render logs/man.obj --wogui --save videos/man --radius 2.5
# python -m kiui.render logs/man_revani.obj --wogui --save videos/man_revani --radius 2.5
# python -m kiui.render logs/man_guofeng.obj --wogui --save videos/man_guofeng --radius 2.5
