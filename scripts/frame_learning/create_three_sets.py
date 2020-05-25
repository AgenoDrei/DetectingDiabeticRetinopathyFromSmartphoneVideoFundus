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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

FILE_COL = 'image'
CLASS_COL = 'level'
CLASS_THRES = 1
FILE_EXT = '.png'

def run(input_path, output_path, labels_path, val_size):
    df = pd.read_csv(labels_path)
    files = os.listdir(input_path)
    df[FILE_COL] = df[FILE_COL].astype(str)
    print(len(df.loc[df['level'] > 3, :]))
    df.loc[df['level'] > 3, 'level'] = 3
    X, y = df[FILE_COL], df[CLASS_COL]
    
    df_train, df_tmp = train_test_split(df, stratify=df[CLASS_COL], test_size=val_size*2)
    df_val, df_test= train_test_split(df_tmp, test_size=0.5, stratify=df_tmp[CLASS_COL])

    for s, name in zip([df_train, df_val, df_test], ['train', 'val', 'test']):
        for row in s.itertuples():
            prefix = 'pos' if row[2] > 1 else 'neg'
            [copy(join(input_path, f), join(output_path, name, prefix, f)) for f in files if row[1] in f]
        s['level'] = s.level.astype(int)
        s.to_csv(join(output_path, name, f'labels_{name}.csv'), index=False)


if __name__ == '__main__':
    a = argparse.ArgumentParser(description='Split Paxos trainings data into train/val set using stratified shuffling')
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--output", help="absolute path to output folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--valsize", help="Percentage of validation set size", type=float, default=0.1)
    args = a.parse_args()
    
    os.mkdir(args.output)
    sets = ('train', 'val', 'test')
    for s in sets:
        os.mkdir(join(args.output, s))
        os.mkdir(join(args.output, s, 'pos'))
        os.mkdir(join(args.output, s, 'neg'))

    run(args.input, args.output, args.labels, args.valsize)
