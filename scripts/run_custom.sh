export CUDA_VISIBLE_DEVICES=7

python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="a cool man with black hair" save_path=man.obj text_dir=True front_dir=+z 
python main.py --config configs/revani.yaml mesh=data2/napoleon.obj prompt="a cool man with black hair" save_path=man_revani.obj text_dir=True front_dir=+z 
python main.py --config configs/guofeng.yaml mesh=data2/napoleon.obj prompt="a cool man with black hair" save_path=man_guofeng.obj text_dir=True front_dir=+z 

python -m kiui.render logs/man.obj --wogui --save videos/man --radius 2.5
python -m kiui.render logs/man_revani.obj --wogui --save videos/man_revani --radius 2.5
python -m kiui.render logs/man_guofeng.obj --wogui --save videos/man_guofeng --radius 2.5

python main.py --config configs/base.yaml mesh='4c865c5df17741f7be87392d37fa31eb' prompt="a beautiful girl with pink hair" save_path=girl.obj text_dir=True
python main.py --config configs/revani.yaml mesh='4c865c5df17741f7be87392d37fa31eb' prompt="a beautiful girl with pink hair" save_path=girl_revani.obj text_dir=True
python main.py --config configs/guofeng.yaml mesh='4c865c5df17741f7be87392d37fa31eb' prompt="a beautiful girl with pink hair" save_path=girl_guofeng.obj text_dir=True

python -m kiui.render logs/girl.obj --wogui --save videos/girl --radius 2.5
python -m kiui.render logs/girl_revani.obj --wogui --save videos/girl_revani --radius 2.5
python -m kiui.render logs/girl_guofeng.obj --wogui --save videos/girl_guofeng --radius 2.5
