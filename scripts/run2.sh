export CUDA_VISIBLE_DEVICES=4

python main.py --config configs/base.yaml mesh=data2/napoleon.obj prompt="A photo of Napoleon Bonaparte" save_path=napoleon.obj text_dir=True 
python main.py --config configs/base.yaml mesh=data2/nascar.obj prompt="a black car" save_path=car.obj text_dir=True 
python main.py --config configs/base.yaml mesh=data2/bunny.obj prompt="a grey bunny" save_path=bunny.obj text_dir=False 
python main.py --config configs/base.yaml mesh=data2/cow.obj prompt="a white and black cow" save_path=cow.obj text_dir=False front_dir=+x
python main.py --config configs/base.yaml mesh=data2/teapot.obj prompt="a chinese teapot made of blue and white porcelain" save_path=teapot.obj text_dir=False 

python main.py --config configs/base.yaml mesh=data/compass.obj prompt="a compass" save_path=compass.obj text_dir=True front_dir=+y2
python main.py --config configs/revani.yaml mesh=data/ambulance.obj prompt="an ambulance" save_path=ambulance.obj text_dir=True front_dir=-x

python main.py --config configs/base.yaml mesh=../gg3d/logs/csm_luigi.obj prompt="luigi" save_path=luigi.obj text_dir=True