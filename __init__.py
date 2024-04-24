# all code in this directory is adapted from the official v-jepa repo:
# https://github.com/facebookresearch/jepa
# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License 

import sys, os

saycam_path = '/N/project/baby_vision_benchmark/github/saycam-video-models/'
# saycam_path = os.path.join(project_root, 'jepa')
util_path = os.path.join(saycam_path, 'util')

if saycam_path not in sys.path:
    sys.path.append(saycam_path)
    
if util_path not in sys.path:
    sys.path.append(util_path) 
