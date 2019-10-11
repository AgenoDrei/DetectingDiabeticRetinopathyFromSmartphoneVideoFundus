import os
import cv2
import numpy as np

####################################
######### HELPER METHODS ###########
####################################

def load_images(path='./C001R_Cut', img_type='jpg'):
    frames = []
    paths = [f for f in os.listdir(path) if f.endswith(img_type)]
    print(f'INFO> Found this frames in folder {path}: {paths}')

    for p in paths:
        image_path = os.path.join(os.getcwd(), path, p)
        image = cv2.imread(image_path)
        frames.append(image)

    return frames

def load_image(path:str):
    print(f'INFO> Loading picture {path}')

    image_path = os.path.join(os.getcwd(), path)
    image = cv2.imread(image_path)
    return image


def show_image(data:np.array, name:str='Single Image', w:int=800, h:int=600, time=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, data)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def show_image_row(data:list, name:str='Image stack', time=0):
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

################### Image functions ##################

def get_retina_mask(img:np.array, min_radius=450, enhance_contrast=False, contrast_clip_limit=-1) -> np.array:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 3, minRadius=min_radius, maxRadius=600,
                               param1=75, param2=75)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            #print('Circles: ', (x, y, r))
            if y < img.shape[0] / 3 or y > img.shape[0] / 3 * 2 or x < img.shape[1] / 3 or x > img.shape[1] / 3 * 2:
                continue
            radius = 500 if r > 500 else r
            mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
            cv2.circle(mask, (x, y), r, (255, 255, 255,), thickness=-1)
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif enhance_contrast and contrast_clip_limit != -1:
        img = enhance_contrast_image(img, clip_limit=contrast_clip_limit)
        return get_retina_mask(img)
    raise Exception('No circular lense detected!')


def enhance_contrast_image(img:np.array, clip_limit=3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    #cl = cv2.equalizeHist(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)

    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final