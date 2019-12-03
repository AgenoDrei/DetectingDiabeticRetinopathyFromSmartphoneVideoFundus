import argparse
import os
import sys
import time
from skvideo import io
sys.path.append('./')
sys.path.append('../include/')
import features as ft
import time_wrap as tw
import utils as utl
import time_wrap as tw
import numpy as np
from os.path import join
import cv2
import joblib as job
import shutil
import multiprocessing
import tqdm as t


SUBFOLDER_FRAMES = 'frames'
SUBFOLDER_PROCESSED = 'processed'
SUBFOLDER_RESULTS = 'results'
NUM_HARALICK_FEATURES = 84
FRAMES_PER_SNIPPET = 20

@tw.profile
def run(input_path, output_path, model_path, fps=10, majority=0.65):
    num_cpus = multiprocessing.cpu_count()
    init(output_path)
    # extract snippets
    # split snippets into frames
    frames = extract_frames(input_path, frames_per_second=fps)
    frames = job.Parallel(n_jobs=num_cpus)(job.delayed(preprocess_frames)(frame) for frame in t.tqdm(frames))

    pipeline = job.load(model_path)
    extractor = ft.FeatureExtractor(haralick_dist=4, clip_limit=4.0, hist_size=[8, 3, 3])

    X_test = create_dataset(frames)
    X_test = extractor.transform(X_test)
    print(X_test.shape)
    y_pred = pipeline.predict(X_test)

    snippets = majority_vote(frames, y_pred, majority=majority)

    print(f'VPRO> SVM found {len(snippets) // FRAMES_PER_SNIPPET} retina snippets')
    print(f'VPRO> Writing frames to {join(output_path, SUBFOLDER_RESULTS)}')

    write_snippets_to_disk(snippets, output_path, name=os.path.splitext(os.path.basename(input_path))[0], fps=fps)

def init(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    os.mkdir(join(output_path, SUBFOLDER_FRAMES))
    os.mkdir(join(output_path, SUBFOLDER_PROCESSED))
    os.mkdir(join(output_path, SUBFOLDER_RESULTS))


@tw.profile
def extract_frames(image_path: str, frames_per_second: int = 10) -> list:
    assert os.path.exists(image_path)
    time_between_frames = 1000 / frames_per_second
    frames = []
    count = 0

    vidcap = cv2.VideoCapture(image_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    prev = -1
    success, image = vidcap.read()
    success = True
    print(f'SNIP> Extracting {frame_count // fps * frames_per_second} frames from {image_path}')
    while True:
        grabbed = vidcap.grab()
        if grabbed:
            time_s = vidcap.get(cv2.CAP_PROP_POS_MSEC) // time_between_frames
            if time_s > prev:
                frames.append(vidcap.retrieve()[1])
            prev = time_s
        else:
            break
        #vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * time_between_frames))  # added this line
        #success, image = vidcap.read()
        #frames.append(image)
        #count = count + 1
    return frames[:len(frames)//FRAMES_PER_SNIPPET * FRAMES_PER_SNIPPET]


def preprocess_frames(img):
    if (type(img) != np.ndarray and type(img) != np.memmap) or img is None:
        return None

    img_enh = utl.enhance_contrast_image(img, clip_limit=4, tile_size=12)
    mask, circle = utl.get_retina_mask(img_enh)
    if circle[2] == 0:
        return None

    img = cv2.bitwise_and(img, mask)
    img = utl.crop_to_circle(img, circle)
    return img


@tw.profile
def create_dataset(frames: list):
    print(f'VPRO> Reading {len(frames)} into dataset')
    frames = [np.random.randint(0, 256, (850, 850, 3), dtype=np.uint8) if frames[i] is None or frames[i].size == 0 else frames[i] for i in range(1, len(frames))]
    return frames


def majority_vote(frames, y_pred, majority: float = 0.65):
    relevant_frames = []
    for i in range(0, len(y_pred), FRAMES_PER_SNIPPET):
        pos_votes = np.sum(y_pred[i:i+FRAMES_PER_SNIPPET])
        if pos_votes >= majority * FRAMES_PER_SNIPPET:
            relevant_frames.extend(frames[i:i+FRAMES_PER_SNIPPET])

    return relevant_frames


def write_snippets_to_disk(snippets, output_path, name:str = 'Output', fps: int = 10):
    snippets = [np.random.randint(0, 256, (850, 850, 3)) if snippets[i] is None or snippets[i].size == 0 else snippets[i] for i in range(1, len(snippets))]
    max_size = max(snippets, key=lambda img: img.shape[0]).shape[0]
    snippets = [utl.pad_image_to_size(snip, (max_size, max_size)) for snip in snippets]

    for i in range(0, len(snippets), FRAMES_PER_SNIPPET):
        #out = cv2.VideoWriter(join(output_path, SUBFOLDER_RESULTS, f'{name}_{i}.a'), cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, size)
        out = io.FFmpegWriter(join(output_path, SUBFOLDER_RESULTS, f'{name}_{i}.mp4'),
                              inputdict={'-r': str(fps), '-s': f'{max_size}x{max_size}'},
                              outputdict={'-vcodec': 'libx264', '-crf': '0', '-preset':'veryslow'})
        for frame in snippets[i:i+FRAMES_PER_SNIPPET]:
            out.writeFrame(frame[:,:,[2, 1, 0]])
        out.close()

    #[cv2.imwrite(join(output_path, SUBFOLDER_RESULTS, f'{i}.jpg'), frame) for i, frame in enumerate(snippets)]



if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to useful images")
    a.add_argument("--pipeline", help="path to SVM pretrained model")
    a.add_argument("--fps", help="Number of frames extracted per second", default=10, type=int)
    a.add_argument("--majority", help="Percentage of frames that have to register as retina", default=0.65, type=float)
    args = a.parse_args()
    print('VPRO> ', args)

    run(args.input, args.output, args.pipeline, fps=args.fps, majority=args.majority)