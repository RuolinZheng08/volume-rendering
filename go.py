# go.c
import sys
sys.path.append('python-packages') # pip installed modules

import numpy as np
import nrrd
from tqdm import tqdm

# my modules
from utils import construct_context
from convolution import Convolution
from ray import Ray

def main():
    params_dict = parse_args(sys.argv)
    context = construct_context(params_dict)
    num_rows, num_cols = context.camera.img_plane_size
    img_out = np.empty((4, num_rows, num_cols)) # 4 for RGBA
    if not context.num_threads: # 0 or unspecified
        ray = Ray()
        convolution = Convolution()
        for col in tqdm(range(num_cols), position=0):
            for row in tqdm(range(num_rows), position=1, leave=False):
                result = ray.go(row, col, convolution, context)
                img_out[:, row, col] = result
    else: # multithread
        pass
    fpath_out = params_dict['fpath_out']
    # TODO: write headers as well
    nrrd.write(fpath_out, img_out)

def parse_args(args):
    params_dict = {}
    # params for setting up the camera
    params_dict['params_camera'] = {}
    params_dict['params_camera']['fr'] = np.array([[6, 12, 5]]).T
    params_dict['params_camera']['at'] = np.array([[0, 0, 0]]).T
    params_dict['params_camera']['up'] = np.array([[0, 0, 1]]).T
    params_dict['params_camera']['near_clip'] = -2.3
    params_dict['params_camera']['far_clip'] = 2.3
    params_dict['params_camera']['field_of_view'] = 14
    # params_dict['params_camera']['img_plane_size'] = (320, 280)
    params_dict['params_camera']['img_plane_size'] = (30, 20)
    params_dict['params_camera']['ortho'] = False

    params_dict['unit_step'] = 0.03
    params_dict['plane_sep'] = 0.03
    params_dict['num_threads'] = 0
    params_dict['outside_val'] = np.nan
    params_dict['alpha_near_one'] = 1

    # file paths
    params_dict['fpath_volume'] = 'utils/cube.nrrd'
    params_dict['fpath_lut'] = 'utils/lut.nrrd'
    params_dict['fpath_light'] = 'utils/rgb.txt'
    params_dict['fpath_out'] = 'utils/cube-rgb-py.nrrd'

    return params_dict

if __name__ == '__main__':
    main()
