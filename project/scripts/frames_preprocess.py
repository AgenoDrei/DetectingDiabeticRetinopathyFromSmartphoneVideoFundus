import os
import argparse
import cv2
from joblib import Parallel, delayed
import sys
import multiprocessing
import numpy as np
sys.path.append('/data/simon/Code/MasterThesis/project/include')
import utils as utl


def run(input_path, output_path):
    paths = [f for f in os.listdir(input_path)]
    print(f'PRE> Found {len(paths)} images in folder {input_path}: {paths}')

    num_cpus = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cpus)(delayed(preprocess_images)(os.path.join(input_path, p)) for p in paths)  # n_jobs = number of processes
    results = [ret for ret in results if type(ret) == np.ndarray]

    [cv2.imwrite(f'{output_path}/{paths[i][:-4]}_{i}.jpg', results[i]) for i in range(len(results))]
    return output_path


def preprocess_images(image_path):
    print(f'PRE> Extracting lens, enhancing contrast and cropping image {image_path}')
    img = cv2.imread(image_path)
    if type(img) != np.ndarray:
        return None

    img_enh = utl.enhance_contrast_image(img, clip_limit=3.5, tile_size=12)
    mask, circle = utl.get_retina_mask(img_enh)
    if circle[2] == 0:
        return None

    img = cv2.bitwise_and(img, mask)
    img = utl.crop_to_circle(img, circle)
    return img


if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to images")
    args = a.parse_args()
    print('PRE> ', args)

    run(args.input, args.output)

    # paths = [f for f in os.listdir(args.input)]
    # print(f'UTIL> Found {len(paths)} images in folder {args.input}: {paths}')
    #
    # num_cpus = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cpus)(delayed(preprocess_images)(os.path.join(args.input, p)) for p in paths)  # n_jobs = number of processes
    # results = [ret for ret in results if type(ret) == np.ndarray]
    #
    # [cv2.imwrite(f'{args.output}/{i}.jpg', results[i]) for i in range(len(results))]


