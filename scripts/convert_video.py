import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='logs')
parser.add_argument('--out', type=str, default='videos')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

files = glob.glob(os.path.join(args.dir, '*.obj'))
for f in files:
    name = os.path.basename(f)
    if os.path.exists(os.path.join(args.out, name.replace('.obj', '.mp4'))) and not args.overwrite:
        print(f'[INFO] skip {name}')
        continue
    print(f'[INFO] process {name}')
    os.system(f"python -m kiui.render {f} --wogui --save_video {os.path.join(args.out, name.replace('.obj', '.mp4'))} --radius 2.5 --elevation=-15")
    os.system(f"python -m kiui.render {f} --wogui --save {os.path.join(args.out, name.replace('.obj', ''))} --radius 2.5 --elevation=-15")
