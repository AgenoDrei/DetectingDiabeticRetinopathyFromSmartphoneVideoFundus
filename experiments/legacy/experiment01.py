import numpy as np
import cv2
from utils import  load_image, show_image, load_images, show_image_row, print_progress_bar
import sys
import os


def run():
    images = load_images()
    show_image_row(images, name='Raw images')

    #images = sharpen(images)
    #show_image_row(images, name='Sharpened')

    images = remove_glare(images)
    show_image_row(images, name='Glare Removed')

    images = enhance_contrast(images)
    show_image_row(images, name='Contrast Enhanced')

    panorama = stitch_images(images)
    if panorama is not None: show_image(panorama, name="Stitched image")


def remove_glare(images:list):
    for i in range(len(images)):
        images[i] = remove_glare_image(images[i])
        print_progress_bar(i + 1, len(images), prefix="Removing glare")
    return images


def enhance_contrast(images:list):
    for i in range(len(images)):
        images[i] = enhance_contrast_image(images[i])
        print_progress_bar(i + 1, len(images), prefix="Enhancing contrast")
    return images


def sharpen(images:list):
    for i in range(len(images)):
        images[i] = sharpen_image(images[i])
        print_progress_bar(i + 1, len(images), prefix="Removing glare")
    return images


def remove_glare_image(img:np.array, saturation_thres=180, color_thres=200):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))  # split into HSV components
    nonSat = s < saturation_thres  # Find all pixels that are not very saturated
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
    v2 = v.copy()
    v2[nonSat == 0] = 0

    glare = v2 > color_thres
    glare = cv2.dilate(glare.astype(np.uint8), disk)
    glare = cv2.dilate(glare.astype(np.uint8), disk)

    return cv2.inpaint(img, glare, 5, cv2.INPAINT_NS)


def enhance_contrast_image(img:np.array, cl=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    #cl = cv2.equalizeHist()
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def sharpen_image(img:np.array):
    #result = cv2.GaussianBlur(img, (3, 3), 0)
    #result = cv2.addWeighted(result, 1.5, img, -0.5, 0)

    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])  # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    return sharpened


def stitch_images(images: list):
    print("INFO> stitching images")
    stitcher = cv2.Stitcher_create(mode=0)
    (status, stitched) = stitcher.stitch(images)

    if status == 0:
        return stitched
    else:
        print(f"[INFO] image stitching failed. Status: {status}")
        return None


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
