import sys
import matplotlib.pyplot as plt
import argparse
import cv2
from os.path import join
from utils import show_image, load_images, show_image_row, enhance_contrast_image, \
    get_retina_mask, get_hsv_colors, show_means, float2gray, print_progress_bar
import os
import unglarer as ug

#NUM_CLUSTERS = 5
#MODEL = '../models/gmm_model_4.mod'
#PATH_SRC = '/data/simon/ownCloud/Data/Reflection Dataset/raw_images/'
#PATH_RES = '~/Data/glare'
#ARTIFACT_THRESHOLD = 0.05
#MATRIX_TYPE = cv2.ml.EM_COV_MAT_GENERIC

def run(input_path, output_path, model_path):
    unglarer: ug.GlareRemover = ug.GlareRemover(model_path=model_path, masked_class=3)

    images = load_images('/data/simon/ownCloud/Data/Reflection Dataset/raw_images/', img_type='png')
    images2 = load_images('./C001R_Cut/', img_type='png')

    images_subset = [images[i] for i in range(0, len(images))]
    images_subset.extend(images2)
    images_subset = [enhance_contrast_image(img, clip_limit=3.5, tile_size=12) for img in images_subset]
    images_subset = [cv2.bitwise_and(img, get_retina_mask(img)[0]) for img in images_subset]
    unglarer.set_training_data(images_subset)

    #unglarer.show_training_data()
    #unglarer.train()
    unglarer.show_means()

    masked_images = []
    files = os.listdir(input_path)
    test_images = [cv2.imread(join(input_path, f)) for f in files]
    for img in test_images:
        mask = unglarer.get_glare_mask(img, show_mask=False, joined_prob=False)
        percentage = unglarer.get_glare_percentage(img, mask)
        masked_images.append((cv2.bitwise_and(img, cv2.cvtColor(mask, code=cv2.COLOR_GRAY2BGR)), percentage))

    #[show_image(img, name=f'Masked images - {percentage:.2f}%') for img, percentage in masked_images]
    [cv2.imwrite(join(output_path, name), f) for name, f in zip(files, masked_images)]



'''
Experimenting GMM
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    parser = argparse.ArgumentParser(description='Script to remove glare from images')
    parser.add_argument('--input', '-i', help='Absolute path to input folder')
    parser.add_argument('--output', '-o', help='Absolute path to output folder')
    parser.add_argument('--model', help='Absolute path to GMM model used for prediction')
    args = parser.parse_args()

    run(args.input, args.output, args.model)
