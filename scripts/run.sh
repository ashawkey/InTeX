export CUDA_VISIBLE_DEVICES=7

python main.py --config configs/revani.yaml mesh=data/dragon.obj camera_path=side prompt="a red pet dragon with fire patterns" save_path=dragon_fire.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data/dragon.obj camera_path=side prompt="a green pet dragon with grass patterns" save_path=dragon_grass.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data/dragon.obj camera_path=side prompt="a blue pet dragon with ice patterns" save_path=dragon_ice.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data/dragon.obj camera_path=side prompt="a pink dragon with cherry patterns" save_path=dragon_cherry.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data/dragon.obj camera_path=side prompt="a robotic dragon with gear and screw patterns" save_path=dragon_robot.obj text_dir=True 

python main.py --config configs/revani.yaml mesh=data2/catpet.obj camera_path=side prompt="a red pet cat with fire patterns" save_path=cat_fire.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/catpet.obj camera_path=side prompt="a green pet cat with grass patterns" save_path=cat_grass.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/catpet.obj camera_path=side prompt="a blue pet cat with ice patterns" save_path=cat_ice.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/catpet.obj camera_path=side prompt="a pink cat with cherry patterns" save_path=cat_cherry.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/catpet.obj camera_path=side prompt="a robotic cat with gear and screw patterns" save_path=cat_robot.obj text_dir=True 

python main.py --config configs/revani.yaml mesh=data2/foxpet.obj camera_path=side prompt="a red pet fox with fire patterns" save_path=fox_fire.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/foxpet.obj camera_path=side prompt="a green pet fox with grass patterns" save_path=fox_grass.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/foxpet.obj camera_path=side prompt="a blue pet fox with ice patterns" save_path=fox_ice.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/foxpet.obj camera_path=side prompt="a pink fox with cherry patterns" save_path=fox_cherry.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/foxpet.obj camera_path=side prompt="a robotic fox with gear and screw patterns" save_path=fox_robot.obj text_dir=True 