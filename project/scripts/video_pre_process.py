import argparse
import os
import sys
sys.path.append('./')
import frame_extract as fe
import frames_preprocess as fp
from os.path import join
import numpy as np
import cv2
import mahotas as mt
import joblib as job
from sklearn import preprocessing, metrics, svm
import shutil

def run(input_path, output_path, model_path, scaler_path):
    init(output_path)
    frames_path = fe.extract_images(input_path, join(output_path, 'frames'), input_path.split('/')[-1], time_between_frames=500)
    processed_frames_path = fp.run(frames_path, join(output_path, 'processed'))

    model: svm.SVC = job.load(model_path)
    scaler: preprocessing.MinMaxScaler = job.load(scaler_path)

    X_test, index = create_dataset(processed_frames_path)
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)

    print(f'SVM found {len(y_pred) - sum(y_pred)} retina frames')
    print(f'Writing frames to {join(output_path, "results")}')
    [os.rename(join(processed_frames_path, index[i]), join(output_path, 'results', index[i])) for i, y in enumerate(y_pred) if y == 0]


def init(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    os.mkdir(join(output_path, 'frames'))
    os.mkdir(join(output_path, 'processed'))
    os.mkdir(join(output_path, 'results'))


def extract_feature_vector(img: np.array) -> np.array:
    assert type(img) == np.ndarray

    channels = cv2.split(img)
    features = []
    for c in channels:
        textures: np.array = mt.features.haralick(c)
        ht_mean = textures.mean(axis=0)
        ht_range = np.ptp(textures, axis=0)
        f = np.hstack((ht_mean, ht_range))
        features.append(f)
    return np.hstack(features)


def create_dataset(processed_frames_path):
    feats = []
    files = os.listdir(processed_frames_path)
    print(f'Reading {len(files)} from {processed_frames_path} into dataset')

    files = [f for f in files if os.stat(join(processed_frames_path, f)).st_size > 0]           # removes empty images in image list

    for p in files:
        feat = extract_feature_vector(cv2.imread(join(processed_frames_path, p)))
        feats.append(feat)

    return feats, files



if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to useful images")
    a.add_argument("--model", help="path to SVM pretrained model")
    a.add_argument("--scaler", help="path to Scaler from the pretrained model")
    args = a.parse_args()
    print(args)

    run(args.input, args.output, args.model, args.scaler)