import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import sys
import os


def run():
    images = load_images()
    show_image_row(images)

    images = enhance_contrast(images)
    show_image_row(images)

    images = remove_glare(images)
    show_image_row(images)

    panorama = stitch_images(images)
    if panorama is not None: show_image(panorama, name="Stitched image")


def remove_glare(images:list):
    for i in range(len(images)):
        images[i] = remove_glare_image(images[i])
        printProgressBar(i + 1, len(images), prefix="Removing glare")
    return images


def remove_glare_image(img:np.array, saturation_thres=100, color_thres=200):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))  # split into HSV components
    nonSat = s < saturation_thres  # Find all pixels that are not very saturated
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    nonSat = cv2.erode(nonSat.astype(np.uint8), disk)
    v2 = v.copy()
    v2[nonSat == 0] = 0

    glare = v2 > color_thres
    glare = cv2.dilate(glare.astype(np.uint8), disk)
    glare = cv2.dilate(glare.astype(np.uint8), disk)

    return cv2.inpaint(img, glare, 5, cv2.INPAINT_NS)


def enhance_contrast(images:list):
    for i in range(len(images)):
        images[i] = enhance_contrast_image(images[i])
        printProgressBar(i + 1, len(images), prefix="Enhancing contrast")
    return images


def enhance_contrast_image(img:np.array, cl=4.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(16, 16))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def stitch_images(images: list):
    print("INFO> stitching images...")
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    if status == 0:
        return stitched
    else:
        print(f"[INFO] image stitching failed. Status: {status}")
        return None

####################################
######### HELPER METHODS ###########
####################################

def load_images(path='./C001R_Cut'):
    frames = []
    paths = [f for f in os.listdir(path) if f.endswith('jpg')]
    print(f'INFO>Found this frames in folder {path}: {paths}')

    for p in paths:
        image_path = os.path.join(os.getcwd(), path, p)
        image = cv2.imread(image_path)
        frames.append(image)

    return frames


def show_image(data:np.array, name:str='Image', w:int=800, h:int=600):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, data)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

def show_image_row(data:list, name:str='Image stack'):
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
    show_image(image_row, name=name, h=max_height, w=1600)




def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')

    run()
