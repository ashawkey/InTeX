export CUDA_VISIBLE_DEVICES=1

python gui.py --mesh data/dragon.obj --prompt "a red pet dragon with fire patterns" --save_path dragon_fire.obj --text_dir --wogui
python gui.py --mesh data/dragon.obj --prompt "a white pet dragon with cloud patterns" --save_path dragon_cloud.obj --text_dir --wogui
python gui.py --mesh data/dragon.obj --prompt "a green pet dragon with grass patterns" --save_path dragon_grass.obj --text_dir --wogui
python gui.py --mesh data/dragon.obj --prompt "a blue pet dragon with ice patterns" --save_path dragon_ice.obj --text_dir --wogui
python gui.py --mesh data/dragon.obj --prompt "a mechanical and robotic dragon with gears and screws patterns" --save_path dragon_robo.obj --text_dir --wogui

python gui.py --mesh data2/catpet.obj --prompt "a red pet cat with fire patterns" --save_path cat_fire.obj --text_dir --wogui
python gui.py --mesh data2/catpet.obj --prompt "a white pet cat with cloud patterns" --save_path cat_cloud.obj --text_dir --wogui
python gui.py --mesh data2/catpet.obj --prompt "a green pet cat with grass patterns" --save_path cat_grass.obj --text_dir --wogui
python gui.py --mesh data2/catpet.obj --prompt "a blue pet cat with ice patterns" --save_path cat_ice.obj --text_dir --wogui
python gui.py --mesh data2/catpet.obj --prompt "a mechanical and robotic cat with gears and screws patterns" --save_path cat_robo.obj --text_dir --wogui

python gui.py --mesh data2/foxpet.obj --prompt "a red pet fox with fire patterns" --save_path fox_fire.obj --text_dir --wogui
python gui.py --mesh data2/foxpet.obj --prompt "a white pet fox with cloud patterns" --save_path fox_cloud.obj --text_dir --wogui
python gui.py --mesh data2/foxpet.obj --prompt "a green pet fox with grass patterns" --save_path fox_grass.obj --text_dir --wogui
python gui.py --mesh data2/foxpet.obj --prompt "a blue pet fox with ice patterns" --save_path fox_ice.obj --text_dir --wogui
python gui.py --mesh data2/foxpet.obj --prompt "a mechanical and robotic fox with gears and screws patterns" --save_path fox_robo.obj --text_dir --wogui