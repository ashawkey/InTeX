export CUDA_VISIBLE_DEVICES=0
python gui.py --mesh data2/napoleon.obj --prompt "a statue of napoleon" --save_path napoleon.obj --text_dir --wogui
python gui.py --mesh data2/nascar.obj --prompt "a cool car" --save_path car.obj --text_dir --wogui
python gui.py --mesh data2/bunny.obj --prompt "a cute rabbit" --save_path rabbit.obj --wogui
python gui.py --mesh data2/cow.obj --prompt "a cow" --save_path cow.obj --text_dir --wogui