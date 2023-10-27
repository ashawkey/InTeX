import os
import glob

out = 'videos'
os.makedirs(out, exist_ok=True)

files = glob.glob('logs/*.obj')
for f in files:
    name = os.path.basename(f)
    print(f'[INFO] process {name}')
    os.system(f"python -m kiui.render {f} --wogui --save_video {os.path.join(out, name.replace('.obj', '.mp4'))} --radius 2.5")
    os.system(f"python -m kiui.render {f} --wogui --save videos/{os.path.join(out, name.replace('.obj', ''))} --radius 2.5")
