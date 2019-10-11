from utils import  load_image, show_image, load_images, show_image_row, print_progress_bar
import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans

NUM_CLUSTERS = 5
CONTRAST_LIMIT = 2
np.set_printoptions(threshold=sys.maxsize)

def run():
    images = load_images(path='./C001R', img_type='png')
    #images = [enhance_contrast_image(images[i]) for i in range(len(images))]
    [detect_circles(img) for img in images]

    image = load_image('/home/simon/Videos/Anomaly Dataset/raw_images/SNAP_00035.png')
    show_image(image, name='Raw')

    #image = enhance_contrast_image(image)
    #show_image(image, name='Contrast')

    detect_circles(image)


def detect_circles(img:np.array) -> np.array:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 3, minRadius=450, maxRadius=600, param1=75, param2=75)
    output = img.copy()

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if y < img.shape[0] / 3 or y > img.shape[0] / 3 * 2 or x < img.shape[1] / 3 or x > img.shape[1] / 3 * 2:
                continue
            r = 500 if r > 500 else r
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
            cv2.circle(mask, (x, y), r, (255, 255, 255, ), thickness=-1)
            show_image_row([output, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], name='circles')
            return mask


def enhance_contrast_image(img:np.array):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CONTRAST_LIMIT, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    #cl = cv2.equalizeHist(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)

    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


'''
Experimenting with circle detection
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()