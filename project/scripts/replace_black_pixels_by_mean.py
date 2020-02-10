import argparse
import cv2
from tqdm import tqdm
import os
import joblib as j
import numpy as np

def run(input_path, output_path):
    files = os.listdir(input_path)
    j.Parallel(n_jobs=-1, verbose=0)(j.delayed(replace_black_pixels_by_mean)(input_path, f, output_path) for f in tqdm(files, total=len(files)))


def replace_black_pixels_by_mean(input_path, img_name, output_path):
    if not os.path.isfile(os.path.join(input_path, img_name)):
        return

    img = cv2.imread(os.path.join(input_path, img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY_INV)
    mean_value = img[thresh != 255].mean(axis=0)
    #std_value =  img[thresh != 255].std(axis=0)
    #img[thresh == 255] = np.clip(np.random.normal(loc=mean_value, scale=std_value), 0, 255)
    img[thresh == 255] = mean_value

    cv2.imwrite(os.path.join(output_path, img_name), img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = argparse.ArgumentParser()
    a.add_argument("--input", "-i", help="absolute path to input folder")
    a.add_argument("--output", "-o", help="absolute path to output folder")
    args = a.parse_args()
    print(args)

    if os.path.exists(args.output):
        os.rmdir(args.output)
    os.mkdir(args.output)

    run(args.input, args.output)