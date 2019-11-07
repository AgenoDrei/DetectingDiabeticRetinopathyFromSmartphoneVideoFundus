import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils import show_image, load_images, show_image_row, enhance_contrast_image, \
    get_retina_mask, get_hsv_colors, show_means, float2gray, print_progress_bar

import unglarer as ug

#NUM_CLUSTERS = 5
MODEL = 'gmm_model_4.mod'
#ARTIFACT_THRESHOLD = 0.05
#MATRIX_TYPE = cv2.ml.EM_COV_MAT_GENERIC

def run():
    unglarer: ug.GlareRemover = ug.GlareRemover(model_path=MODEL, masked_class=3)

    images = load_images('/data/simon/ownCloud/Data/Reflection Dataset/raw_images/', img_type='png')
    images2 = load_images('./C001R/', img_type='png')

    images_subset = [images[i] for i in range(0, len(images))]
    images_subset.extend(images2)
    images_subset = [enhance_contrast_image(img, clip_limit=3.5, tile_size=12) for img in images_subset]
    images_subset = [cv2.bitwise_and(img, get_retina_mask(img)[0]) for img in images_subset]
    unglarer.set_training_data(images_subset)

    #unglarer.show_training_data()
    #unglarer.train()
    unglarer.show_means()

    masked_images = []
    for img in unglarer.get_training_data():
        mask = unglarer.get_glare_mask(img, show_mask=False, joined_prob=False)
        percentage = unglarer.get_glare_percentage(img, mask)
        masked_images.append((cv2.bitwise_and(img, cv2.cvtColor(mask, code=cv2.COLOR_GRAY2BGR)), percentage))

    [show_image(img, name=f'Masked images - {percentage:.2f}%') for img, percentage in masked_images]


'''
Experimenting GMM
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
