from utils import  load_image, show_image, load_images, show_image_row, print_progress_bar, enhance_contrast_image
import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans

NUM_CLUSTERS = 5
CONTRAST_LIMIT = 2
np.set_printoptions(threshold=sys.maxsize)

def run():
    images = load_images(path='./C001R', img_type='png')
    images = [enhance_contrast_image(images[i], clip_limit=4) for i in range(len(images))]
    show_image_row(images, 'Contrast')
    [detect_circles(img) for img in images]

    image = load_image('/home/simon/Videos/Anomaly Dataset/raw_images/SNAP_00035.png')
    show_image(image, name='Raw')

    #image = enhance_contrast_image(image)
    #show_image(image, name='Contrast')

    detect_circles(image)


def detect_circles(img:np.array, radius_reduction=20) -> np.array:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)
    output = img.copy()
    circle = None

    #detect small lens
    small_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 8, minRadius=400, maxRadius=500, param1=90, param2=90)
    if small_circles is not None:
        small_circles = np.round(small_circles[0, :]).astype("int")
        small_circles = [(x, y, r) for (x, y, r) in small_circles if img.shape[0] / 3 < y < img.shape[0] / 3 * 2 and img.shape[1] / 3 < x < img.shape[1] / 3 * 2]
        circle = sorted(small_circles, key=lambda xyr: xyr[2])[0]

    if circle is None:                   #detect large lens
        large_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 8, minRadius=500, maxRadius=600, param1=90, param2=90)
        if large_circles is not None:
            large_circles = np.round(large_circles[0, :]).astype("int")
            large_circles = [(x, y, r) for (x, y, r) in large_circles if
                             img.shape[0] / 3 < y < img.shape[0] / 3 * 2 and img.shape[1] / 3 < x < img.shape[
                                 1] / 3 * 2]
            circle = sorted(large_circles, key=lambda xyr: xyr[2])[0]

    if circle is not None:
        (x, y, r) = circle
        r -= radius_reduction
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        cv2.circle(mask, (x, y), r, (255, 255, 255,), thickness=-1)
        show_image_row([output, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], name='circles')
        return mask

    print('INFO> No mask found')
    return None


'''
Experimenting with circle detection
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()