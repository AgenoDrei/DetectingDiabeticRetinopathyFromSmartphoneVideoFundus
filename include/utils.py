import os
from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skvideo import io
import time_wrap as tw
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import exposure, img_as_ubyte
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys


"""
Helper methods useful to manipulate images and videos
"""


def load_images(path: str = './C001R_Cut', img_type: str = 'jpg') -> list:
    frames = []
    paths = [f for f in os.listdir(path) if f.endswith(img_type)]
    print(f'UTIL> Found {len(paths)} frames in folder {path}: {paths}')

    for p in paths:
        image_path = os.path.join(os.getcwd(), path, p)
        image = cv2.imread(image_path)
        frames.append(image)

    return frames


def load_image(path: str) -> np.ndarray:
    print(f'UTIL> Loading picture {path}')

    image_path = os.path.join(os.getcwd(), path)
    image = cv2.imread(image_path)
    return image


def show_image(data: np.ndarray, name: str = 'Single Image', w: int = 1200, h: int = 900, time: int = 0) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, data)
    cv2.waitKey(time)
    cv2.destroyAllWindows()


def show_image_row(data: list, name: str = 'Image stack', time: int = 0) -> None:
    max_height: int = 0
    acc_width: int = 0
    for img in data:
        max_height = img.shape[0] if img.shape[0] > max_height else max_height
        acc_width += img.shape[1]

    conc_img = np.zeros(shape=[max_height, acc_width, 3], dtype=np.uint8)
    dups = []
    for img in data:
        delta_height = max_height - img.shape[0]
        top, bottom = delta_height // 2, delta_height - (delta_height // 2)

        duplicate = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
        dups.append(duplicate)
    image_row = np.concatenate(dups, axis=1)
    show_image(image_row, name=name, h=max_height, w=1600, time=time)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def float2gray(img: np.array) -> np.array:
    return np.uint8(cv2.normalize(img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))


def pad_image_to_size(img: np.ndarray, pref_size: tuple) -> np.ndarray:
    pref_size = (int(pref_size[0]), int(pref_size[1]))
    if pref_size[0] == img.shape[0] and pref_size[1] == img.shape[1]:
        return img

    horizontal_pad = (pref_size[1] - img.shape[1]) // 2
    vertical_pad = (pref_size[0] - img.shape[0]) // 2

    padded_img = np.zeros((pref_size[0], pref_size[1], 3))
    padded_img[vertical_pad:vertical_pad + img.shape[0], horizontal_pad:horizontal_pad + img.shape[1]] = img

    # padded_img = np.pad(img, [(vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad)], mode='constant')
    # if padded_img.shape[0] < pref_size[0]:
    #     padded_img = np.pad(padded_img, [(0, 1), (0, 0)], mode='constant')
    # elif padded_img.shape[1] < pref_size[1]:
    #     padded_img = np.pad(padded_img, [(0, 0), (0, 1)], mode='constant')

    # print(f'UTILS> Pref: {pref_size}, Padded: {padded_img.shape}')
    return padded_img


################### Image functions ##################
def get_retina_mask(img: np.ndarray, radius_reduction: int = 20, hough_param: int = 75, min_radius = 300, speedup = 4) -> (np.ndarray, tuple):
    h, w = img.shape[0] // speedup, img.shape[1] // speedup
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dsize=(w, h))
    img_blur = cv2.medianBlur(gray, 3)
    
    min_radius = min_radius // speedup
    mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    circle = None

    # detect lens
    small_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, gray.shape[0] / 8, minRadius=min_radius,
                                     maxRadius=min_radius+(300 // speedup), param1=hough_param, param2=60)
    if small_circles is not None:
        small_circles = np.round(small_circles[0, :]).astype("int")
        small_circles = [(x, y, r) for (x, y, r) in small_circles if
                         gray.shape[0] / 3 < y < gray.shape[0] / 3 * 2 and gray.shape[1] / 3 < x < gray.shape[1] / 3 * 2]
        circle = sorted(small_circles, key=lambda xyr: xyr[2])[0] if len(small_circles) > 0 else None

    if circle is not None:
        (x, y, r) = circle
        
        r -= radius_reduction
        cv2.circle(mask, (x*speedup, y*speedup), r*speedup, (255, 255, 255,), thickness=-1)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (x*speedup, y*speedup, r*speedup)
    else:
        print('UTIL> No mask found')
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (0, 0, 0)

def get_retina_mask_old(img: np.ndarray, radius_reduction: int = 20, hough_param: int = 75, min_radius = 200) -> (np.ndarray, tuple):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    circle = None

    # detect small lens
    small_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 8, minRadius=min_radius,
                                     maxRadius=min_radius+75, param1=hough_param, param2=60)
    if small_circles is not None:
        small_circles = np.round(small_circles[0, :]).astype("int")
        small_circles = [(x, y, r) for (x, y, r) in small_circles if
                         img.shape[0] / 3 < y < img.shape[0] / 3 * 2 and img.shape[1] / 3 < x < img.shape[
                             1] / 3 * 2]
        circle = sorted(small_circles, key=lambda xyr: xyr[2])[0] if len(small_circles) > 0 else None
    else:
        large_circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0] / 8, minRadius=min_radius+75,
                                         maxRadius=min_radius+150, param1=hough_param, param2=40)
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


def enhance_contrast_image(img: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)
    # cl = cv2.equalizeHist(l)
    ca = clahe.apply(a)
    cb = clahe.apply(b)

    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def crop_to_circle(img: np.ndarray, circle) -> np.ndarray:
    x, y, r = circle
    return img[y - r:y + r, x - r:x + r, :]


def show_means(means: np.ndarray, weights) -> None:
    show_strip = np.zeros((100, means.shape[0] * 100, means.shape[1]))
    progress = 0
    for i, mean in enumerate(means):
        start, stop = int(progress), int(progress + weights[0, i] * 100 * means.shape[0])
        show_strip[0:100, start:stop, :] = mean
        progress += weights[0, i] * 100 * means.shape[0]

    show_strip = np.uint8(show_strip)
    # print(show_strip.shape)
    show_image(cv2.cvtColor(show_strip, cv2.COLOR_HSV2BGR))


def get_hsv_colors(n: int) -> np.ndarray:
    colors = np.zeros((n, 3), dtype=np.uint8)
    hue = np.arange(0, 180, 180 / n)
    colors[:, 0] = hue
    colors[:, 1] = colors[:, 2] = 255
    return colors


def plot_historgram_one_channel(img: np.ndarray) -> None:
    hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist, 'g.')
    plt.xlim([0, 255])
    plt.ylim(0, max(hist))
    plt.show()


def do_five_crop(img: np.ndarray, height: int, width: int, state: int = 0) -> np.ndarray:
    if state == 0:
        top = (img.shape[0] - height) // 2
        left = (img.shape[1] - width) // 2
    elif state == 1:
        top = int(img.shape[0] * 0.05)
        left = int(img.shape[1] * 0.05)
    elif state == 2:
        top = int(img.shape[0] * 0.45)
        left = int(img.shape[1] * 0.05)
    elif state == 3:
        top = int(img.shape[0] * 0.05)
        left = int(img.shape[1] * 0.45)
    else:
        top = int(img.shape[0] * 0.45)
        left = int(img.shape[1] * 0.45)
    return img[top:top + height, left:left + width]


def draw_image_grid(images: list, clahe=False) -> None:
    if clahe:
        [exposure.equalize_adapthist(s, clip_limit=0.02) for s in images]
    size = int(np.ceil(np.sqrt(len(images))))
    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111, nrows_ncols=(size, size), axes_pad=0.1)

    for ax, im in zip(grid, images):
        ax.set_axis_off()
        ax.imshow(im)
    plt.axis('off')
    plt.show()
    plt.close(fig)


########################## Video functions ###############################

def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(frames_dir, os.path.splitext(video_filename)[0] + "_{:004d}.jpg".format(frame // every))  # create the save path 
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def extract_video_frames(video_path, frames_dir, overwrite=False, frames_per_second=10, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """
    every = 30 / frames_per_second

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame, also handles case chunk_size < total

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            pass

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames


@tw.profile
def extract_keyframes_ffmpeg(video_path: str, output_path: str) -> None:
    """
    Extract all keyframes from a video file and save them to disk
    :param video_path: Absolute path to the input video
    :param output_path: Absolute path where all keyframes will be saved
    """
    assert os.path.exists(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_data = io.vreader(video_path, outputdict={'-vf': 'select=eq(pict_type\,PICT_TYPE_I)', '-vsync': 'vfr'})
    cnt = 0
    for kframe in video_data:
        cv2.imwrite(join(output_path, f'{video_name}_{cnt:03d}.png'), cv2.cvtColor(kframe, code=cv2.COLOR_RGB2BGR))
        cnt += 1
    print(f'EXTK> Extracted {cnt} keyframes from {os.path.basename(video_path)} to {output_path}')
