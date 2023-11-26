python main.py --config configs/revani.yaml mesh='215b317cabfa42c48fad08d8b9a8fb8d' prompt="cartoon fox" save_path=fox.obj text_dir=True
python main.py --config configs/revani.yaml mesh=data_texfusion/casa.obj prompt="minecraft house, bricks, rock, grass, stone" save_path=casa.obj 
python main.py --config configs/revani.yaml mesh=data_texfusion/casa.obj prompt="Portrait of greek-egyptian deity hermanubis, lapis skin and gold clothing" save_path=deity.obj text_dir=True



python main.py --config configs/revani.yaml mesh='2f14ac56a5a84286bbc8eab9110dbc1e' prompt="a girl with brown hair, full body" save_path=girlbody.obj text_dir=True retex=True
python main.py --config configs/revani.yaml mesh='490a8417cac946899eac86fba72cc210' prompt="a girl with pink hair, full body, cat ears" save_path=girlbody.obj text_dir=True retex=True
