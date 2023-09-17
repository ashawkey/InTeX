import pandas as pd
import os

# run all selected uids
uids = [
    # '523838ecb26b404d948b393a3dcc439e', # sundae
    # '40d50989fec1460f8838b608d999ccd0', # pizza
    # 'f8e13d5694464e8581907dde27bb59c8', # pizza 2
    # 'fc7cdaabeb7749ec92d5fe3e71d97478', # apple
    # 'fa2c41a7a6c84fcb871a24016fa9a932', # doughnut
    # '242aca8c33d64b09a339353585be366e', # hamburger
    # 'd9ecc99108a643ab887296cb986be937', # hamburger 2
    # 'e1b22565fc6942a2866bdf8b51cf567a', # tomato
    # 'e403051d77f84c1ab359caa0866106b7', # icecream
    # '5736985d6deb4af0a4e2c0ba885f72f8', # icecream 2
    # 'fc63df9772a4474ab070fd765bfbadd9', # cup
    # 'ff993f2a369240ec94bf52612c964bb8', # pumpkin
    # 'e1ca50858dd94a10a94b7ac2ca07c69e', # bread
    # '37ae05c23fe34e3199ee6d2175fbe23f', # bread 2
    # 'faef9fe5ace445e7b2989d1c1ece361c', # shiba
    # 'cf0c100d3ee44371ada969cfba5af34f', # whelk
    # 'ca1f7d67dda34644a50729b95fe17d18', # owl
    # 'eb219212147f4d84b88f8e103af8ea10', # frog
    # 'c246565eb927410486c7cf27b138a2e2', # penguin
    # '6fd942297dc34b6cb12dca4be2710b80', # cat
    # 'd2d0c7fbaadc43b2bacd15c51714265a', # cat 2
    # 'f8f16346224045b6914b83e02c35e0ae', # rat
    # 'fc655111af5b49bf84722affc3ddba00', # fish
    '9a6bd7edfc414b0c964d1a12e16a2cf1', # fish 2
    # '911b309008d24829b65f72fd913b91b7', # fish 3
    # 'fd6a48cc80194eb3839cca1975cf3f06', # bear
    # '4a763a1c211044089b1315f9f025b027', # starfish
    # 'fa059adf40334e23beef2f49d05cd46f', # pistol
    # 'fc24cb34811b42f2976e413939cbb08b', # hydrant
    # 'cfc2172e2c414b708d8eadcd38d27b0c', # mushroom
    # 'bad47b128a554f748af6ed0d0370671f', # mushroom 2
    # 'df325cfe1fa24108bdfef58ba7e88b3a', # rabbit
    # 'dd0d65a5c6f643a2af2e930c640d45f4', # wind mill
    # 'b4f65f62d5c543e39fc89aa3a08dd1e9', # wind mill 2
    # 'ef682dfee3c946fabebefb616c3bf8c0', # lamp
    # 'fd57d303020b432aa920bef6887607e3', # violin
    # 'f0478a8bb6b4419886c2817001509c42', # snowman
    # 'fb2e16d216ca4fda948a073ad5ee26a5', # bike
    # 'fc9cc06615084298b4c0c0a02244f356', # piano
]

cap3d = pd.read_csv('Cap3D_automated_Objaverse_no3Dword.csv', header=None)

for uid in uids:
    prompt = cap3d[cap3d[0] == uid][1].values[0]
    print(f'===== processing {uid}: {prompt} =====')
    cmd = f'python main.py --config configs/revani.yaml mesh={uid} prompt="{prompt}" save_path={uid}.obj interactive=True'
    os.system(cmd)