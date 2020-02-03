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
    df = pd.read_csv(labels_path)
    df['image'] = df.image.astype(str)
    df_val = pd.DataFrame(columns=df.columns)
    df_train = pd.DataFrame(columns=df.columns)
    #print(df.level.value_counts())

    X, y = df['image'], df['level']
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
    split = next(splitter.split(X, y))

    for idx in split[0]: #idx of train set
        df_train = df_train.append(df.iloc[idx].to_dict(), ignore_index=True)
        copy_files(df.iloc[idx, 0], input_path, df.iloc[idx, 1], 'train')
    for idx in split[1]: #idx of validation set
        df_val = df_val.append(df.iloc[idx].to_dict(), ignore_index=True)
        copy_files(df.iloc[idx, 0], input_path,df.iloc[idx, 1], 'val')
            
    df_val.to_csv(join(input_path, 'labels_val.csv'), index=False)
    df_train.to_csv(join(input_path, 'labels_train.csv'), index=False)


def copy_files(id, input_path, level, set_id):
    prefix = 'pos' if level == 1 else 'neg'
    frames = os.listdir(os.path.join(input_path, f'{prefix}_snips_frames'))
    
    job.Parallel(n_jobs=-1, verbose=0)(job.delayed(copy)(join(input_path, f'{prefix}_snips_frames', f), join(input_path, set_id, prefix, os.path.basename(f))) for f in frames if id in f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--valsize", help="Percentage of validation set size", type=float, default=0.1)
    #a.add_argument("--output", help="absolute path to output folder")
    args = a.parse_args()
    
    os.mkdir(join(args.input, 'train', 'pos'))
    os.mkdir(join(args.input, 'train', 'neg'))
    os.mkdir(join(args.input, 'val', 'pos'))
    os.mkdir(join(args.input, 'val', 'neg'))

    run(args.input, args.labels, args.valsize)





