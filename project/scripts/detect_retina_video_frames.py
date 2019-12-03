import argparse
import os
import sys
sys.path.append('./')
sys.path.append('../include/')
import frame_extract as fe
import frames_preprocess as fp
import features as ft
import time_wrap as tw
from os.path import join
import cv2
import joblib as job
import shutil
import multiprocessing


SUBFOLDER_FRAMES = 'frames'
SUBFOLDER_PROCESSED = 'processed'
SUBFOLDER_RESULTS = 'results'
NUM_HARALICK_FEATURES = 84

@tw.profile
def run(input_path, output_path, model_path):
    init(output_path)
    frames_path = fe.extract_images(input_path, join(output_path, SUBFOLDER_FRAMES), input_path.split('/')[-1], time_between_frames=750)
    processed_frames_path = fp.run(frames_path, join(output_path, SUBFOLDER_PROCESSED))

    pipeline = job.load(model_path)
    extractor = ft.FeatureExtractor(haralick_dist=4, clip_limit=4.0, hist_size=[8, 3, 3])

    X_test, index = create_dataset(processed_frames_path)
    X_test = extractor.transform(X_test)
    y_pred = pipeline.predict(X_test)

    print(f'VPRO> SVM found {sum(y_pred)} retina frames')
    print(f'VPRO> Writing frames to {join(output_path, SUBFOLDER_RESULTS)}')
    [os.rename(join(processed_frames_path, index[i]), join(output_path, SUBFOLDER_RESULTS, index[i])) for i, y in enumerate(y_pred) if y == 1]


def init(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    os.mkdir(join(output_path, SUBFOLDER_FRAMES))
    os.mkdir(join(output_path, SUBFOLDER_PROCESSED))
    os.mkdir(join(output_path, SUBFOLDER_RESULTS))


def create_dataset(path):
    files = os.listdir(path)
    print(f'VPRO> Reading {len(files)} from {path} into dataset')

    files = [f for f in files if os.stat(join(path, f)).st_size > 0]  # removes empty images in image list

    num_cpus = multiprocessing.cpu_count()
    x = job.Parallel(n_jobs=num_cpus)(job.delayed(cv2.imread)(join(path, f)) for f in files)

    return x, files


# def create_dataset(processed_frames_path):
#     feats = []
#     files = os.listdir(processed_frames_path)
#     print(f'Reading {len(files)} from {processed_frames_path} into dataset')
#
#     files = [f for f in files if os.stat(join(processed_frames_path, f)).st_size > 0]           # removes empty images in image list
#
#     for p in files:
#         feat = extract_feature_vector(cv2.imread(join(processed_frames_path, p)))
#         feats.append(feat)
#
#     return np.array(feats), files


if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to useful images")
    a.add_argument("--pipeline", help="path to SVM pretrained model")
    #a.add_argument("--transformer", help="path to Scaler from the pretrained model")
    args = a.parse_args()
    print('VPRO> ', args)

    run(args.input, args.output, args.pipeline)