export CUDA_VISIBLE_DEVICES=4

# data from other paper
python main.py --config configs/revani.yaml mesh=data2/nascar.obj prompt="a black car" save_path=car.obj text_dir=True 
python main.py --config configs/revani.yaml mesh=data2/cow.obj prompt="a white and black cow" save_path=cow.obj text_dir=False front_dir=+x
python main.py --config configs/revani.yaml mesh=data2/teapot.obj prompt="a chinese teapot made of blue and white porcelain" save_path=teapot.obj text_dir=False 
# data from objaverse
python main.py --config configs/revani.yaml mesh=data_objaverse/floral_vase.glb prompt="a white vase with red dots decoration" save_path=vase.obj text_dir=True front_dir=-y
python main.py --config configs/revani.yaml mesh=data_objaverse/clock.glb prompt="a beautiful clock" save_path=clock.obj text_dir=True front_dir=+z
python main.py --config configs/revani.yaml mesh=data_objaverse/extra_chocolate_marshmallow_cupcake.glb prompt="a chocolate cake" save_path=cake.obj text_dir=False front_dir=+z
python main.py --config configs/revani.yaml mesh=data_objaverse/forest_house.glb prompt="a forest house" save_path=house.obj text_dir=False front_dir=+z retex=True
python main.py --config configs/revani.yaml mesh=data_objaverse/guitar.glb prompt="a cool guitar" save_path=guitar.obj text_dir=False front_dir=+x1 retex=True
python main.py --config configs/revani.yaml mesh=data_objaverse/handpainted_watercolor_cake.glb prompt="a strawberry cake" save_path=cake2.obj text_dir=False front_dir=+z
python main.py --config configs/revani.yaml mesh=data_objaverse/leather_bag.glb prompt="a leather bag" save_path=bag.obj text_dir=False front_dir=-y
python main.py --config configs/revani.yaml mesh=data_objaverse/strawberry_cake.glb prompt="a strawberry cake" save_path=cake3.obj text_dir=False front_dir=+z
python main.py --config configs/revani.yaml mesh=data_objaverse/sofa.glb prompt="a leather sofa" save_path=sofa.obj text_dir=False front_dir=-x3