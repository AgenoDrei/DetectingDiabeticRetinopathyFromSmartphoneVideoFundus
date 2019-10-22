from utils import  load_image, show_image, load_images, show_image_row, print_progress_bar, get_retina_mask, get_hsv_colors
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

NUM_CLUSTERS = 5
CONTRAST_LIMIT = 4
np.set_printoptions(threshold=sys.maxsize)

def run():
    image = load_image('/home/simon/Videos/Anomaly Dataset/raw_images/SNAP_00035.png')
    image2 = load_image('./C001R_Cut/C001R04.jpg')
    show_image_row([image, image2])

    image, image2 = enhance_contrast_image(image), enhance_contrast_image(image2)
    show_image_row([image, image2])

    image_mask = get_retina_mask(image)
    image = cv2.bitwise_and(image, image_mask)

    image, image2 = cv2.medianBlur(image, 5), cv2.medianBlur(image2, 5)
    show_image_row([image, image2])

    cluster_image(image)
    visualize_color_space(image)
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

    center[0:NUM_CLUSTERS] = get_hsv_colors(NUM_CLUSTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    show_image_row([res2, img], name='Result clustering')
    return img


def get_features(img:np.array):
    #descriptor = cv2.xfeatures2d.SIFT_create()
    descriptor = cv2.ORB_create()
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


def visualize_color_space(img:np.array) -> None:
    show_image(img, name='Pre-colorspace calc')
    r, g, b = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

'''
Experimenting with clustering / anomaliy detection
'''
if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()