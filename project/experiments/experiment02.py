import numpy as np
import cv2
import sys
from utils import  load_image, show_image, enhance_contrast_image, show_image_row

RED_AMPLIFICATION_COEFF = 1.1


def run():
    image = load_image(path='./C001R_Cut/C001R04.jpg')
    show_image(image)

    show_channels(image)
    show_channels(image, show_hsv_channels=False)

    image_cl = enhance_contrast_image(image, clip_limit=4.0)
    image_cl_blur = enhance_contrast_image(cv2.GaussianBlur(image, (7, 7), 0))
    image_clg = clahe_green_channel(image, clip_limit=4.0, back_merge=False)
    image_clg_blur = clahe_green_channel(cv2.GaussianBlur(image, (7, 7), 0), clip_limit=4.0, back_merge=False)
    show_image_row([image_cl, image_cl_blur, image_clg, image_clg_blur], name='Green Clahe enhanced')

    #image = remove_glare_image(image)
    #show_image(image)


    #image = amplify_red_channel(image, coeff=RED_AMPLIFICATION_COEFF)
    #show_image(image, name=f'Red channel {RED_AMPLIFICATION_COEFF}')

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


def show_channels(img:np.array, show_hsv_channels=False) -> None:
    if show_hsv_channels:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img)
    show_image_row(channels, 'channels')
    if show_hsv_channels:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def amplify_red_channel(img:np.array, coeff=1.1) -> np.array:
    img = np.float32(img.copy())
    img[:, :, 0] *= (1 / coeff)
    img[:, :, 2] *= (1 / coeff)
    img[:, :, 1] *= coeff
    img = np.clip(img, 0, 256)
    return np.uint8(img)


def clahe_green_channel(img: np.array, clip_limit: float = 2.0, back_merge: bool = False):
    green_channel = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    green_enhanced = clahe.apply(green_channel)
    if back_merge:
        return cv2.merge([img[:, :, 0], green_enhanced, img[:, :, 2]])
    return cv2.cvtColor(green_enhanced, code=cv2.COLOR_GRAY2BGR)

'''
Experimenting with a single image for preprocessing parameters
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
