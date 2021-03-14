# go.c
import sys
sys.path.append('python-packages') # pip installed modules

import argparse
from types import SimpleNamespace

import numpy as np
import nrrd
from tqdm import tqdm

# threading
import threading
from concurrent.futures import ThreadPoolExecutor

# my modules
from utils import construct_context
from convolution import Convolution
from ray import Ray

# globals
global_context = None
global_img_out = None
global_row, global_col = 0, 0
global_mutex = threading.Lock()

def main():
    params_dict = parse_args() # using argparse.ArgumentParser

    # context could be shared among threads using nonlocal
    global global_context, global_img_out
    global_context = construct_context(params_dict)
    num_rows, num_cols = global_context.camera.img_plane_size
    global_img_out = np.empty((4, num_rows, num_cols)) # 4 for RGBA

    if not global_context.num_threads: # 0 or unspecified
        ray = Ray()
        convolution = Convolution()
        for col in tqdm(range(num_cols), position=0):
            for row in tqdm(range(num_rows), position=1, leave=False):
                result = ray.go(row, col, convolution, global_context)
                global_img_out[:, row, col] = result

    else: # multithread, 1 only incurs locking overhead
        num_threads = global_context.num_threads
        thread_args = []
        for tid in range(num_threads):
            targ = SimpleNamespace()
            targ.tid = tid
            # private
            targ.ray = Ray()
            targ.convolution = Convolution()
            thread_args.append(targ)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            executor.map(thread_function, thread_args)

    fpath_out = params_dict['fpath_out']
    # TODO: write headers as well
    nrrd.write(fpath_out, global_img_out)

def thread_func(args):
    global global_context, global_img_out
    num_rows, num_cols = context.camera.img_plane_size
    while True:
        # lock
        global global_mutex
        with global_mutex:
            global global_row, global_col
            # cache values locally
            row = global_row
            col = global_col
            if row < num_rows - 1:
                global_row += 1
            else:
                # reset horizontal and increment vertical for other threads to see
                global_row = 0
                global_col += 1
        # unlock
        if col == num_cols:
            break # done
        result = args.ray.go(row, col, args.convolution, context)
        global_img_out[:, row, col] = result

def parse_args():
    parser = argparse.ArgumentParser(description='Volume rendering and ray marching.')
    parser.add_argument('-i', dest='input', required=True)
    parser.add_argument('-fr', type=float, nargs=3, action='append', required=True)
    parser.add_argument('-at', type=float, nargs=3, action='append', required=True)
    parser.add_argument('-up', type=float, nargs=3, action='append', required=True)
    parser.add_argument('-nc', type=float, required=True)
    parser.add_argument('-fc', type=float, required=True)
    parser.add_argument('-fov', type=float, required=True)
    parser.add_argument('-us', type=float, required=True)
    parser.add_argument('-s', type=float, required=True)
    parser.add_argument('-lut', required=True)
    parser.add_argument('-lit', required=True)
    parser.add_argument('-sz', type=int, nargs=2, required=True)
    parser.add_argument('-nt', type=int, required=True)
    parser.add_argument('-o', dest='output', required=True)
    parser.add_argument('-ortho', action='store_true')

    args = parser.parse_args()

    # return value
    params_dict = {}
    # params for setting up the camera
    params_dict['params_camera'] = {}
    params_dict['params_camera']['fr'] = np.array(args.fr).T
    params_dict['params_camera']['at'] = np.array(args.at).T
    params_dict['params_camera']['up'] = np.array(args.up).T
    params_dict['params_camera']['near_clip'] = args.nc
    params_dict['params_camera']['far_clip'] = args.fc
    params_dict['params_camera']['field_of_view'] = args.fov
    params_dict['params_camera']['img_plane_size'] = args.sz
    params_dict['params_camera']['ortho'] = args.ortho

    params_dict['unit_step'] = args.us
    params_dict['plane_sep'] = args.s
    params_dict['num_threads'] = args.nt
    params_dict['outside_val'] = np.nan # TODO
    params_dict['alpha_near_one'] = 1 # TODO

    # file paths
    params_dict['fpath_volume'] = args.input
    params_dict['fpath_lut'] = args.lut
    params_dict['fpath_light'] = args.lit
    params_dict['fpath_out'] = args.output

    return params_dict

if __name__ == '__main__':
    main()
