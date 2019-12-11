import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

####################################
######### HELPER METHODS ###########
####################################


def load_images(path: str = './C001R_Cut', img_type:str = 'jpg') -> list:
    frames = []
    paths = [f for f in os.listdir(path) if f.endswith(img_type)]
    print(f'UTIL> Found {len(paths)} frames in folder {path}: {paths}')

    for p in paths:
        image_path = os.path.join(os.getcwd(), path, p)
        image = cv2.imread(image_path)
        frames.append(image)

    return frames


def load_image(path: str) -> np.array:
    print(f'UTIL> Loading picture {path}')

    image_path = os.path.join(os.getcwd(), path)
    image = cv2.imread(image_path)
    return image


def show_image(data: np.array, name: str = 'Single Image', w: int = 1200, h: int = 900, time: int = 0) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, data)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def show_image_row(data:list, name: str = 'Image stack', time: int = 0) -> None:
    max_height:int = 0
    acc_width: int = 0
    for img in data:
        max_height = img.shape[0] if img.shape[0] > max_height else max_height
        acc_width += img.shape[1]

    conc_img = np.zeros(shape=[max_height, acc_width, 3], dtype=np.uint8)
    dups =  []
    for img in data:
        delta_height = max_height - img.shape[0]
        top, bottom = delta_height // 2, delta_height - (delta_height // 2)

        duplicate = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
        dups.append(duplicate)
    image_row = np.concatenate(dups, axis=1)
    show_image(image_row, name=name, h=max_height, w=1600, time=time)


def print_progress_bar (iteration, total, prefix ='', suffix ='', decimals = 1, length = 100, fill ='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def float2gray(img: np.array) -> np.array:
    return np.uint8(cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))


def pad_image_to_size(img: np.ndarray, pref_size: tuple) -> np.ndarray:
    if pref_size[0] == img.shape[0] and pref_size[1] == img.shape[1]:
        return img

    horizontal_pad = (pref_size[1] - img.shape[1]) // 2
    vertical_pad = (pref_size[0] - img.shape[0]) // 2

    padded_img = np.zeros((pref_size[0], pref_size[1], 3))
    padded_img[vertical_pad:vertical_pad+img.shape[0], horizontal_pad:horizontal_pad+img.shape[1]] = img

    # padded_img = np.pad(img, [(vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad)], mode='constant')
    # if padded_img.shape[0] < pref_size[0]:
    #     padded_img = np.pad(padded_img, [(0, 1), (0, 0)], mode='constant')
    # elif padded_img.shape[1] < pref_size[1]:
    #     padded_img = np.pad(padded_img, [(0, 0), (0, 1)], mode='constant')

    #print(f'UTILS> Pref: {pref_size}, Padded: {padded_img.shape}')
    return padded_img


################### Image functions ##################
def get_retina_mask(img:np.array, radius_reduction: int = 20, hough_param:int = 75) -> (np.array, tuple):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    circle = None

    # detect small lens
    small_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 8, minRadius=400,
                                     maxRadius=470, param1=hough_param, param2=60)
    if small_circles is not None:
        small_circles = np.round(small_circles[0, :]).astype("int")
        small_circles = [(x, y, r) for (x, y, r) in small_circles if
                         img.shape[0] / 3 < y < img.shape[0] / 3 * 2 and img.shape[1] / 3 < x < img.shape[
                             1] / 3 * 2]
        circle = sorted(small_circles, key=lambda xyr: xyr[2])[0] if len(small_circles) > 0 else None
    else:
        large_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 8, minRadius=470,
                                         maxRadius=570, param1=hough_param, param2=40)
        if large_circles is not None:
            large_circles = np.round(large_circles[0, :]).astype("int")
            large_circles = [(x, y, r) for (x, y, r) in large_circles if
                             img.shape[0] / 3 < y < img.shape[0] / 3 * 2 and img.shape[1] / 3 < x < img.shape[
                                 1] / 3 * 2]
            circle = sorted(large_circles, key=lambda xyr: xyr[2])[0] if len(large_circles) > 0 else None

    if circle is not None:
        (x, y, r) = circle
        r -= radius_reduction
        cv2.circle(mask, (x, y), r, (255, 255, 255,), thickness=-1)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), circle
    else:
        print('UTIL> No mask found')
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (0, 0, 0)


def enhance_contrast_image(img:np.array, clip_limit: float = 3.0, tile_size: int = 8) -> np.array:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)
    #cl = cv2.equalizeHist(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)

    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def crop_to_circle(img: np.array, circle) -> np.array:
    x, y, r = circle
    return img[y - r:y + r, x - r:x + r, :]


def show_means(means: np.array, weights) -> None:
    show_strip = np.zeros((100, means.shape[0] * 100, means.shape[1]))
    progress = 0
    for i, mean in enumerate(means):
        start, stop = int(progress), int(progress + weights[0, i] * 100 * means.shape[0])
        show_strip[0:100, start:stop, :] = mean
        progress += weights[0, i] * 100 * means.shape[0]

    show_strip = np.uint8(show_strip)
    #print(show_strip.shape)
    show_image(cv2.cvtColor(show_strip, cv2.COLOR_HSV2BGR))


def get_hsv_colors(n: int) -> np.array:
    colors = np.zeros((n, 3), dtype=np.uint8)
    hue = np.arange(0, 180, 180 / n)
    colors[:, 0] = hue
    colors[:, 1] = colors[:, 2] = 255
    return colors


def plot_historgram_one_channel(img: np.array) -> None:
    hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist, 'g.')
    plt.xlim([0, 255])
    plt.ylim(0, max(hist))
    plt.show()
