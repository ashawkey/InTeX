import os
import glob

out = 'videos'
os.makedirs(out, exist_ok=True)

files = glob.glob('logs/*.obj')
for f in files:
    name = os.path.basename(f)
    # first stage model, ignore
    if name.endswith('_mesh.obj'): 
        continue
    print(f'[INFO] process {name}')
    os.system(f"python -m kiui.render {f} --wogui --save_video {os.path.join(out, name.replace('.obj', '.mp4'))} --radius 2")