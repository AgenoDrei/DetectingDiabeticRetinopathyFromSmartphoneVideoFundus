import numpy as np
import cv2
import sys
from utils import  load_image, show_image, enhance_contrast_image, show_image_row

np.set_printoptions(threshold=sys.maxsize)

def run() -> None:
    #image = load_image(path='./C001R_Cut/C001R08.jpg')
    image = load_image('/home/simon/ownCloud/Data/SFBI Studie/high_quality/SL0601.WM0289.P1.04.H.VQPQ2105.PNG')
    show_image(image)

    bg = extract_background_pixel(image, grid_size=150, t=1.)
    retinal_enhancement(image, bg)


def extract_background_pixel(image:np.array, grid_size:int=200, t:float=1.0) -> np.array:
    print('INFO> Initial image dimensions: ', image.shape)
    img = cv2.split(image)[1]     #extract green channel
    top = int((img.shape[0] - img.shape[1]) / 2)
    img = img[top:top+img.shape[1], :]
    print('INFO> Cropped image dimensions: ', img.shape)

    #img = cv2.equalizeHist(img)
    #img = cv2.GaussianBlur(img, (7, 7), -1)
    img = np.float64(img)
    img = (img - img.min()) / (img.max() - img.min())

    show_image(np.uint8(img*256), 'Raw')

    step_size = img.shape[0] / grid_size
    mean_img = np.zeros((grid_size, grid_size), dtype=np.float64)
    dev_img = np.zeros((grid_size, grid_size), dtype=np.float64)

    for y in np.arange(step_size/2, img.shape[0], step_size):
        for x in np.arange(step_size/2, img.shape[1], step_size):
            sample = img[int(y-step_size/2):int(y+step_size/2), int(x-step_size/2):int(x+step_size/2)]
            sample = sample.flatten()
            #show_image(np.uint8(sample * 256), f'Sample Pos({y}|{x}) Size({step_size})')

            mean, dev = cv2.meanStdDev(sample)
            mean_img[int(y / step_size), int(x / step_size)] = mean.sum()
            dev_img[int(y / step_size), int(x / step_size)] = dev.sum()

    show_image(np.uint8(mean_img*256), name='mean')
    show_image(np.uint8(dev_img*256), name='dev')

    mean_img = cv2.resize(mean_img, img.shape, interpolation=cv2.INTER_CUBIC)
    dev_img = cv2.resize(dev_img, img.shape, interpolation=cv2.INTER_CUBIC)

    print(mean_img.shape, dev_img.shape, img.shape)

    # Avoid dividing by zero in next step
    #dev_img_mask = dev_img == 0
    #dev_img[dev_img_mask] = sys.float_info.min

    dist_img = cv2.subtract(img, mean_img)
    dist_img = cv2.divide(dist_img, dev_img)
    dist_img = np.absolute(dist_img)
    bg_img = np.zeros(img.shape, dtype=np.uint8)
    bg_mask = dist_img <= t
    bg_img[bg_mask] = 255

    show_image(bg_img, name='Background')
    return bg_img

    #for y in range(background_image.shape[0]):
    #    for x in range(background_image.shape[1]):

def retinal_enhancement(img:np.array, bg:np.array) -> None:
    pass


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
    sys.exit(0)