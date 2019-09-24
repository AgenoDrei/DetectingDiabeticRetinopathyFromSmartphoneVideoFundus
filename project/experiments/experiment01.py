import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import sys
import os


def run():
    images = load_images()
    show_image(images[1], w=600, h=800, name='First frame')


def load_images(path='./C001R'):
    frames = []
    paths = [f for f in os.listdir(path) if f.endswith('png')]
    print(f'Found this frames in folder {path}: {paths}')

    for p in paths:
        image_path = os.path.join(os.getcwd(), path, p)
        image = cv2.imread(image_path)
        frames.append(image)

    return frames


def show_image(data, name='Image', w=800, h=600):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, data)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(f'Using python version {sys.version_info}')
    print(f'Using opencv version {cv2.__version__}')

    run()
