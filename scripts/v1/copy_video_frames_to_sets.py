import joblib as job
import argparse
import cv2
import os
import numpy as np
from os.path import join
import pandas as pd
import re
from pathlib import Path
from shutil import copy
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


def run(input_path, labels_path, val_size):
    '''
    Split Paxos trainings data into train and validtation set using stratified shuffeling
    :param input_path: Path to input data (should have folders pos/neg
    :param labels_path: Path to CSV describing images (row 0) and level (row 1)
    :param val_size: Size of validation set
    :return:
    '''
    df = pd.read_csv(labels_path)
    df['image'] = df.image.astype(str)
    df_val = pd.DataFrame(columns=df.columns)
    df_train = pd.DataFrame(columns=df.columns)
    #print(df.level.value_counts())

    X, y = df['image'], df['level']
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
    split = next(splitter.split(X, y))

    for idx in tqdm(split[0], total=len(split[0]), desc='Train data'): #idx of train set
        severity = 1 if df.iloc[idx, 1] > 0 else 0
        df_train = df_train.append({'image': df.iloc[idx, 0], 'level': severity}, ignore_index=True)
        copy_files(df.iloc[idx, 0], input_path, severity, 'train')
    for idx in tqdm(split[1], total=len(split[1]), desc='Val data'): #idx of validation set
        severity = 1 if df.iloc[idx, 1] > 0 else 0
        df_val = df_val.append({'image': df.iloc[idx, 0], 'level': severity}, ignore_index=True)
        copy_files(df.iloc[idx, 0], input_path, severity, 'val')
            
    df_val.to_csv(join(input_path, 'labels_val.csv'), index=False)
    df_train.to_csv(join(input_path, 'labels_train.csv'), index=False)


def copy_files(id, input_path, level, set_id):
    prefix = 'pos' if level == 1 else 'neg'
    frames = os.listdir(os.path.join(input_path, f'{prefix}'))
    
    job.Parallel(n_jobs=-1, verbose=0)(job.delayed(copy)(join(input_path, f'{prefix}', f), join(input_path, set_id, prefix, os.path.basename(f))) for f in frames if id in f)


if __name__ == '__main__':
    a = argparse.ArgumentParser(description='Split Paxos trainings data into train/val set using stratified shuffling')
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--valsize", help="Percentage of validation set size", type=float, default=0.1)
    #a.add_argument("--output", help="absolute path to output folder")
    args = a.parse_args()
    
    os.mkdir(join(args.input, 'train'))
    os.mkdir(join(args.input, 'val'))
    os.mkdir(join(args.input, 'train', 'pos'))
    os.mkdir(join(args.input, 'train', 'neg'))
    os.mkdir(join(args.input, 'val', 'pos'))
    os.mkdir(join(args.input, 'val', 'neg'))

    run(args.input, args.labels, args.valsize)






