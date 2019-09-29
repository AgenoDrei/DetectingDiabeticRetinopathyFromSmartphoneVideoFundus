from utils import  load_image, show_image, load_images, show_image_row, print_progress_bar
import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans

NUM_CLUSTERS = 5
CONTRAST_LIMIT = 3
np.set_printoptions(threshold=sys.maxsize)

def run():
    image = load_image('./C001R_Cut/C001R01.jpg')
    show_image(image)

    image = enhance_contrast_image(image)
    show_image(image)

    clusters = cluster_image(image)


def cluster_image(img:np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img_data = img.reshape((img.shape[0] * img.shape[1], 3))
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, NUM_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    print(center[2])

    #center[0] = np.array([0, 255, 255])
    #center[1] = np.array([30, 255, 255])
    center[2] = np.array([60, 255, 255])
    #center[3] = np.array([90, 255, 255])
    #center[4] = np.array([120, 255, 255])
    #center[5] = np.array([150, 255, 255])
    #center[6] = np.array([179, 255, 255])

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    show_image_row([res2, img], name='Result clustering')

    #clt = KMeans(n_clusters=NUM_CLUSTERS)
    #clt.fit(img_data)

    #print(clt.labels_, clt.cluster_centers_)



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
Experimenting with clustering / anomaliy detection
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()