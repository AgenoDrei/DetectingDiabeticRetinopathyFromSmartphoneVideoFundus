import numpy as np
import cv2
import sys
from utils import  load_image, show_image

def run():
    image = load_image()
    show_image(image)

    #image = remove_glare_image(image)
    #show_image(image)

    image = enhance_contrast_image(image)
    show_image(image)


def remove_glare_image(img:np.array, saturation_thres=75, value_thres=125):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))  # split into HSV components
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    glare = cv2.inRange(h, 85, 90)
    glare = cv2.dilate(glare.astype(np.uint8), disk)
    s2 = s.copy()
    s2[glare == 0] = 0

    nonSat = s2 < saturation_thres  # Find all pixels that are not very saturated
    nonSat = cv2.erode(nonSat.astype(np.uint8), disk)

    #v2 = v.copy()
    #v2[nonSat == 0] = 0
    #h2 = h.copy()
    #h2[nonSat == 0] = 0

    # glare = v2 > value_thres
    # glare = cv2.dilate(glare.astype(np.uint8), disk)
    # glare = cv2.dilate(glare.astype(np.uint8), disk)
    #
    # glare_green = cv2.inRange(h2, 80, 90)
    # glare_green = cv2.dilate(glare_green.astype(np.uint8), disk)
    # glare_green = cv2.dilate(glare_green.astype(np.uint8), disk)


    #h2[glare_green == 0] = 0
    show_image(glare, name='Green')
    show_image(nonSat, name='Glare green')
    show_image(cv2.inpaint(img, nonSat, 5, cv2.INPAINT_NS))
    #show_image(cv2.inpaint(cv2.inpaint(img, glare, 5, cv2.INPAINT_NS), glare_green, 5, cv2.INPAINT_NS))

    return cv2.inpaint(img, glare_green, 5, cv2.INPAINT_NS)


def enhance_contrast_image(img:np.array, limit=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    #cl = cv2.equalizeHist(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)

    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


'''
Experimenting with a single image for preprocessing parameters
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
