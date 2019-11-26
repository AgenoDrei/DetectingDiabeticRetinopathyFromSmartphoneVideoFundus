import argparse
import os
import sys
sys.path.append('./')
sys.path.append('../include/')
import frame_extract as fe
import frames_preprocess as fp
import features as ft
from os.path import join
import numpy as np
import cv2
import joblib as job
from sklearn import preprocessing, metrics, svm, compose
import shutil
import multiprocessing


SUBFOLDER_FRAMES = 'frames'
SUBFOLDER_PROCESSED = 'processed'
SUBFOLDER_RESULTS = 'results'
NUM_HARALICK_FEATURES = 84


def run(input_path, output_path, model_path, transformer_path):
    init(output_path)
    frames_path = fe.extract_images(input_path, join(output_path, SUBFOLDER_FRAMES), input_path.split('/')[-1], time_between_frames=500)
    processed_frames_path = fp.run(frames_path, join(output_path, SUBFOLDER_PROCESSED))

    model: svm.SVC = job.load(model_path)
    transformer: compose.ColumnTransformer = job.load(transformer_path)

    X_test, index = create_dataset(processed_frames_path)
    X_test = np.array(X_test)
    X_test = transformer.transform(X_test)
    y_pred = model.predict(X_test)

    print(f'SVM found {sum(y_pred)} retina frames')
    print(f'Writing frames to {join(output_path, SUBFOLDER_RESULTS)}')
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
    print(f'Reading {len(files)} from {path} into dataset')

    files = [f for f in files if os.stat(join(path, f)).st_size > 0]  # removes empty images in image list

    num_cpus = multiprocessing.cpu_count()
    results = job.Parallel(n_jobs=num_cpus)(job.delayed(ft.extract_feature_vector)(cv2.imread(join(path, p))) for p in files)

    return results, files


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
    a.add_argument("--model", help="path to SVM pretrained model")
    a.add_argument("--transformer", help="path to Scaler from the pretrained model")
    args = a.parse_args()
    print(args)

    run(args.input, args.output, args.model, args.transformer)