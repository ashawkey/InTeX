python main.py --gpu 0 --config configs/base.yaml mesh=data2/napoleon.obj prompt="A photo of Napoleon Bonaparte" save_path=napoleon.obj text_dir=True 
python main.py --gpu 0 --config configs/base.yaml mesh=data2/nascar.obj prompt="a black car" save_path=car.obj text_dir=True 
python main.py --gpu 0 --config configs/base.yaml mesh=data2/bunny.obj prompt="a grey bunny" save_path=bunny.obj text_dir=False 
python main.py --gpu 0 --config configs/base.yaml mesh=data2/cow.obj prompt="a white and black cow" save_path=cow.obj text_dir=False 
python main.py --gpu 0 --config configs/base.yaml mesh=data2/teapot.obj prompt="a chinese teapot made of blue and white porcelain" save_path=teapot.obj text_dir=False 