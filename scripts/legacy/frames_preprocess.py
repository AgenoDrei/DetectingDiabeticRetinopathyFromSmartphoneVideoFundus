import os
import argparse
from os.path import join

import cv2
from joblib import Parallel, delayed
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('/data/simon/Code/MasterThesis/project/include')
import utils as utl


def run(input_path, output_path, min_radius=300):
    paths = [f for f in os.listdir(input_path)]
    print(f'PRE> Extracting lens, enhancing contrast and cropping image for {len(paths)} images in folder {input_path}: {paths}')

    Parallel(n_jobs=1, backend='loky')(delayed(preprocess_images)(os.path.join(input_path, p), output_path, min_radius) for p in tqdm(paths, total=len(paths), desc='Cropping'))  # n_jobs = number of processes
    return output_path


def preprocess_images(image_path, output_path, min_radius=300):
    img = cv2.imread(image_path)
    if (type(img) != np.ndarray and type(img) != np.memmap) or img is None:
        return

    img_enh = utl.enhance_contrast_image(img, clip_limit=3.5, tile_size=12)
    mask, circle = utl.get_retina_mask(img_enh, min_radius=min_radius)
    if circle[2] == 0:
        return

    img = cv2.bitwise_and(img, mask)
    img = utl.crop_to_circle(img, circle)

    if img is not None and img.size != 0:
        filepath = join(output_path, os.path.splitext(os.path.basename(image_path))[0]) + '.jpg'
        cv2.imwrite(filepath, img)


if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input_path", "-i", help="path to unprocessed images")
    a.add_argument("--output_path", "-o", help="path to images")
    a.add_argument("--min_radius", "-r", help="Define minimal radius of the lens", default=300)
    args = a.parse_args()
    print('PRE> ', args)

    os.mkdir(args.output_path)

    run(args.input_path, args.output_path, min_radius=int(args.min_radius))

    # paths = [f for f in os.listdir(args.input)]
    # print(f'UTIL> Found {len(paths)} images in folder {args.input}: {paths}')
    #
    # num_cpus = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cpus)(delayed(preprocess_images)(os.path.join(args.input, p)) for p in paths)  # n_jobs = number of processes
    # results = [ret for ret in results if type(ret) == np.ndarray]
    #
    # [cv2.imwrite(f'{args.output}/{i}.jpg', results[i]) for i in range(len(results))]


