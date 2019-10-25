import numpy as np
import cv2
import sys

from utils import  load_image, show_image, enhance_contrast_image, show_image_row, float2gray

np.set_printoptions(threshold=sys.maxsize)

def run() -> None:
    pass


'''PCA test for segmentation'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
    sys.exit(0)