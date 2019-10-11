from utils import  load_image, show_image, load_images, show_image_row, print_progress_bar, get_retina_mask
import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans

NUM_CLUSTERS = 6
CONTRAST_LIMIT = 4
np.set_printoptions(threshold=sys.maxsize)

def run():
    image = load_image('/home/simon/Videos/Anomaly Dataset/raw_images/SNAP_00035.png')
    image2 = load_image('./C001R_Cut/C001R04.jpg')
    show_image_row([image, image2])

    image, image2 = enhance_contrast_image(image), enhance_contrast_image(image2)
    show_image_row([image, image2])

    image_mask = get_retina_mask(image)
    image = cv2.bitwise_and(image, cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR))

    image, image2 = cv2.medianBlur(image, 5), cv2.medianBlur(image2, 5)
    show_image_row([image, image2])

    cluster_image(image)
    #cluster_image(image2)

    ft, ft2 = get_features(image), get_features(image2)
    #matcher = create_matcher()
    #matches = matcher.match(ft['ft'], ft2['ft'])
    #matches = sorted(matches, key=lambda x: x.distance)
    #match_img = cv2.drawMatches(image, ft['kp'], image2, ft2['kp'], matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #show_image(match_img, name='Matches')


def cluster_image(img:np.array):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img_data = img.reshape((img.shape[0] * img.shape[1], 3))
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, NUM_CLUSTERS, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    print(center[2])

    center[0] = np.array([0, 255, 255])
    center[1] = np.array([30, 255, 255])
    center[2] = np.array([60, 255, 255])
    center[3] = np.array([90, 255, 255])
    center[4] = np.array([120, 255, 255])
    center[5] = np.array([150, 255, 255])
    #center[6] = np.array([179, 255, 255])

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    #img[np.reshape((label == 2), img.shape[0:2])] = (255, 255, 255)

    show_image_row([res2, img], name='Result clustering')

    #clt = KMeans(n_clusters=NUM_CLUSTERS)
    #clt.fit(img_data)

    #print(clt.labels_, clt.cluster_centers_)
    return img


def get_features(img:np.array):
    descriptor = cv2.xfeatures2d.SIFT_create()
    #descriptor = cv2.ORB_create()
    img_kp = img.copy()
    (kps, features) = descriptor.detectAndCompute(img, None)

    cv2.drawKeypoints(img, kps, img_kp, color=(0, 0, 255))
    show_image(img_kp)
    return {'kp': kps, 'ft': features}


def create_matcher():
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return bf

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