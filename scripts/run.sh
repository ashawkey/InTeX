export CUDA_VISIBLE_DEVICES=1

python gui.py --mesh data/dragon.obj --prompt "a pet dragon with fire patterns" --save_path dragon_fire.obj --text_dir --wogui
python gui.py --mesh data/dragon.obj --prompt "a pet dragon with cloud patterns" --save_path dragon_cloud.obj --text_dir --wogui
python gui.py --mesh data/dragon.obj --prompt "a pet dragon with grass patterns" --save_path dragon_grass.obj --text_dir --wogui
python gui.py --mesh data/dragon.obj --prompt "a pet dragon with ice patterns" --save_path dragon_ice.obj --text_dir --wogui
python gui.py --mesh data/dragon.obj --prompt "a pet dragon, mechanical and robotic style, with gears and screws" --save_path dragon_robo.obj --text_dir --wogui

python gui.py --mesh data2/catpet.obj --prompt "a pet cat with fire patterns" --save_path cat_fire.obj --text_dir --wogui
python gui.py --mesh data2/catpet.obj --prompt "a pet cat with cloud patterns" --save_path cat_cloud.obj --text_dir --wogui
python gui.py --mesh data2/catpet.obj --prompt "a pet cat with grass patterns" --save_path cat_grass.obj --text_dir --wogui
python gui.py --mesh data2/catpet.obj --prompt "a pet cat with ice patterns" --save_path cat_ice.obj --text_dir --wogui
python gui.py --mesh data2/catpet.obj --prompt "a pet cat, mechanical and robotic style, with gears and screws" --save_path cat_robo.obj --text_dir --wogui

python gui.py --mesh data2/foxpet.obj --prompt "a pet fox with fire patterns" --save_path fox_fire.obj --text_dir --wogui
python gui.py --mesh data2/foxpet.obj --prompt "a pet fox with cloud patterns" --save_path fox_cloud.obj --text_dir --wogui
python gui.py --mesh data2/foxpet.obj --prompt "a pet fox with grass patterns" --save_path fox_grass.obj --text_dir --wogui
python gui.py --mesh data2/foxpet.obj --prompt "a pet fox with ice patterns" --save_path fox_ice.obj --text_dir --wogui
python gui.py --mesh data2/foxpet.obj --prompt "a pet fox, mechanical and robotic style, with gears and screws" --save_path fox_robo.obj --text_dir --wogui