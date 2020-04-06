import argparse
import os
import sys

from tqdm import tqdm

sys.path.append('./')
sys.path.append('../include/')
import features as ft
import time_wrap as tw
from os.path import join
import cv2
import joblib as job
import shutil
import utils as utl
import numpy as np

SUBFOLDER_FRAMES = 'frames'
SUBFOLDER_PROCESSED = 'processed'
SUBFOLDER_RESULTS = 'results'
NUM_HARALICK_FEATURES = 84

@tw.profile
def run(input_path, output_path, model_path, fps=10):
    init(output_path)
    pipeline = job.load(model_path)
    extractor = ft.FeatureExtractor(haralick_dist=4, clip_limit=4.0, hist_size=[8, 3, 3])

    utl.extract_video_frames(str(input_path), join(output_path, SUBFOLDER_FRAMES), frames_per_second=fps)

    frame_paths = sorted(os.listdir(join(output_path, SUBFOLDER_FRAMES)), key=lambda f: int(os.path.splitext(f)[0].split('_')[1]))
    features = job.Parallel(n_jobs=-1, verbose=0, timeout=200)(job.delayed(process_video_frame)(output_path, f, extractor) for f in tqdm(frame_paths, total=len(frame_paths)))
    X_test = np.array(features)
    if X_test.shape[0] == 0:
        return

    y_pred = pipeline.predict(X_test)

    print(f'VPRO> SVM found {sum(y_pred)} retina frames')
    #print(f'VPRO> Writing frames to {join(output_path, SUBFOLDER_RESULTS)}')
    [os.rename(join(output_path, SUBFOLDER_PROCESSED, frame_paths[i]), join(output_path, SUBFOLDER_RESULTS, frame_paths[i])) for i, y in enumerate(y_pred) if y == 1]


def init(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    os.mkdir(join(output_path, SUBFOLDER_FRAMES))
    os.mkdir(join(output_path, SUBFOLDER_PROCESSED))
    os.mkdir(join(output_path, SUBFOLDER_RESULTS))


def process_video_frame(output_path, img_path: str, extractor: ft.FeatureExtractor):
    frame = cv2.imread(join(output_path, SUBFOLDER_FRAMES, img_path))
    frame = preprocess_frames(frame, output_path, img_path)
    frame = np.zeros((850, 850, 3), dtype=np.uint8) if frame is None or frame.size == 0 or (type(frame) != np.ndarray and type(frame) != np.memmap) else frame
    feature_vec = extractor.extract_single_feature_vector(frame, extractor.haralick_dist, extractor.hist_size, extractor.clip_limit)
    return feature_vec


def preprocess_frames(img, output_path, img_path):
    if (type(img) != np.ndarray and type(img) != np.memmap) or img is None:
        return None

    img_enh = utl.enhance_contrast_image(img, clip_limit=3.5, tile_size=12)
    mask, circle = utl.get_retina_mask(img_enh)
    if circle[2] == 0:
        return None

    img = cv2.bitwise_and(img, mask)
    img = utl.crop_to_circle(img, circle)
    if img is not None and img.size != 0:
        cv2.imwrite(join(output_path, SUBFOLDER_PROCESSED, img_path), img)
    return img



if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to useful images")
    a.add_argument("--pipeline", help="path to SVM pretrained model")
    #a.add_argument("--transformer", help="path to Scaler from the pretrained model")
    args = a.parse_args()
    print('VPRO> ', args)

    run(args.input, args.output, args.pipeline)