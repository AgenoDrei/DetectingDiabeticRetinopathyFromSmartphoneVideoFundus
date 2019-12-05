import argparse
import os
import time
from skvideo import io
import features as ft
import utils as utl
import time_wrap as tw
import numpy as np
from os.path import join
import cv2
import joblib as job
import shutil
from tqdm import trange


SUBFOLDER_FRAMES = 'frames'
SUBFOLDER_PROCESSED = 'processed'
SUBFOLDER_RESULTS = 'results'
NUM_HARALICK_FEATURES = 84
FRAMES_PER_SNIPPET = 20
BATCH_SIZE = 240


@tw.profile
def run(input_path: str, output_path: str, model_path: str, fps: int = 10, majority: float = 0.65) -> None:
    """
    :param input_path: path to input video
    :param output_path: path to folder for temporary and result files, will overwrite exiting folder
    :param model_path: path to sklearn pipeline (created with experiment08.ipynb)
    :param fps: number of frames that extracted per second of the video
    :param majority: percentage (0.0 to 1.0) of frames that have to show meaningful information
    :return:
    """

    init(output_path)
    pipeline = job.load(model_path)
    extractor = ft.FeatureExtractor(haralick_dist=4, clip_limit=4.0, hist_size=[8, 3, 3])

    extract_frames(input_path, output_path, frames_per_second=fps)

    X_test = np.empty((0, 156), dtype=np.float)
    file_paths = sorted(os.listdir(join(output_path, SUBFOLDER_FRAMES)), key=lambda f: int(os.path.splitext(f)[0]))
    for i in trange(0, len(file_paths), BATCH_SIZE):
        start = time.monotonic()
        # print(f'VPRO> Start: {start:.2f}')
        frames = job.Parallel(n_jobs=-1, verbose=0)(job.delayed(cv2.imread)(join(output_path, SUBFOLDER_FRAMES, f)) for f in file_paths[i:i+BATCH_SIZE])
        # print(f'VPRO> After reading {time.monotonic()-start:.2f}')
        frames = job.Parallel(n_jobs=-1, verbose=0)(job.delayed(preprocess_frames)(frame, output_path, i+j) for j, frame in enumerate(frames))
        # print(f'VPRO> After pp {time.monotonic()-start:.2f}')
        frames = [np.random.randint(0, 256, (850, 850, 3), dtype=np.uint8) if frames[j] is None or frames[j].size == 0 else frames[j] for j in
                  range(len(frames))]
        features = extractor.transform(frames)
        # print(f'VPRO> End: {time.monotonic()-start:.2f}')
        # print(f'VPRO> Batch shape: {features.shape}, cur X_test shape: {X_test.shape}')

        X_test = np.append(X_test, features, axis=0)

    y_pred = pipeline.predict(X_test)

    snippet_idxs = majority_vote(y_pred, majority=majority)

    print(f'VPRO> SVM found {len(snippet_idxs)} retina snippets')
    print(f'VPRO> Writing frames to {join(output_path, SUBFOLDER_RESULTS)}')

    write_snippets_to_disk(snippet_idxs, output_path, name=os.path.splitext(os.path.basename(input_path))[0], fps=fps)


def init(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    os.mkdir(join(output_path, SUBFOLDER_FRAMES))
    os.mkdir(join(output_path, SUBFOLDER_PROCESSED))
    os.mkdir(join(output_path, SUBFOLDER_RESULTS))


@tw.profile
def extract_frames(image_path: str, output_path: str, frames_per_second: int = 10) -> None:
    assert os.path.exists(image_path)
    time_between_frames = 1000 / frames_per_second
    frames = []
    count = 0

    vidcap = cv2.VideoCapture(image_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    prev = -1
    _, image = vidcap.read()
    print(f'SNIP> Extracting {frame_count // fps * frames_per_second} frames from {image_path}')
    while count <= 5000:                                    # Max video size
        grabbed = vidcap.grab()
        if grabbed:
            time_s = vidcap.get(cv2.CAP_PROP_POS_MSEC) // time_between_frames
            if time_s > prev:
                cv2.imwrite(join(output_path, SUBFOLDER_FRAMES, f'{count}.jpg'), vidcap.retrieve()[1])
                count += 1
                #frames.append(vidcap.retrieve()[1])
            prev = time_s
        else:
            break
    #job.Parallel(n_jobs=-1)(job.delayed(cv2.imwrite)(join(output_path, SUBFOLDER_FRAMES, f'{i}.jpg'), frame) for i, frame in
    #                        enumerate(frames[:len(frames)//FRAMES_PER_SNIPPET * FRAMES_PER_SNIPPET]))


def preprocess_frames(img, out_path, idx):
    if (type(img) != np.ndarray and type(img) != np.memmap) or img is None:
        return None

    img_enh = utl.enhance_contrast_image(img, clip_limit=4, tile_size=12)
    mask, circle = utl.get_retina_mask(img_enh)
    if circle[2] == 0:
        return None

    img = cv2.bitwise_and(img, mask)
    img = utl.crop_to_circle(img, circle)
    cv2.imwrite(join(out_path, SUBFOLDER_PROCESSED, f'{idx}.jpg'), img)
    return img


@tw.profile
def create_dataset(frames: list):
    print(f'VPRO> Reading {len(frames)} into dataset')
    frames = [np.random.randint(0, 256, (850, 850, 3), dtype=np.uint8) if frames[i] is None or frames[i].size == 0 else frames[i] for i in
              range(1, len(frames))]
    return frames


def majority_vote(y_pred, majority: float = 0.65) -> list:
    relevant_frames = []
    for i in range(0, len(y_pred), FRAMES_PER_SNIPPET):
        pos_votes = np.sum(y_pred[i:i+FRAMES_PER_SNIPPET])
        if pos_votes >= majority * FRAMES_PER_SNIPPET:
            relevant_frames.append((i, i+FRAMES_PER_SNIPPET))

    return relevant_frames


@tw.profile
def write_snippets_to_disk(idxs, output_path, name:str = 'Output', fps: int = 10):
    for start, end in idxs:
        frames = job.Parallel(n_jobs=-1, verbose=0)(job.delayed(cv2.imread)(join(output_path, SUBFOLDER_PROCESSED, f'{j}.jpg')) for j in range(start, end))
        frames = [np.random.randint(0, 256, (850, 850, 3)) if f is None or f.size == 0 else f for f in frames]
        max_size = max(frames, key=lambda img: img.shape[0]).shape[0]
        avg_size = sum([f.shape[0] for f in frames])/len(frames)
        frames = [utl.pad_image_to_size(f, (max_size, max_size)) for f in frames]

        #out = cv2.VideoWriter(join(output_path, SUBFOLDER_RESULTS, f'{name}_{i}.a'), cv2.VideoWriter_fourcc('H', '2', '6', '4'), fps, size)
        out = io.FFmpegWriter(join(output_path, SUBFOLDER_RESULTS, f'{name}_{start}.mp4'),
                              inputdict={'-r': str(fps), '-s': f'{max_size}x{max_size}'},
                              outputdict={'-vcodec': 'libx264', '-crf': '0', '-preset':'slow'})
        for frame in frames:
            out.writeFrame(frame[:,:,[2, 1, 0]])
        out.close()

    #[cv2.imwrite(join(output_path, SUBFOLDER_RESULTS, f'{i}.jpg'), frame) for i, frame in enumerate(snippets)]



if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to useful images")
    a.add_argument("--pipeline", help="path to SVM pretrained model")
    a.add_argument("--fps", help="Number of frames extracted per second", default=10, type=int)
    a.add_argument("--majority", help="Percentage of frames that have to register as retina", default=0.6, type=float)
    args = a.parse_args()
    print('VPRO> ', args)

    run(args.input, args.output, args.pipeline, fps=args.fps, majority=args.majority)