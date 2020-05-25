import joblib as job
import argparse
import cv2
import os
import numpy as np
from os.path import join
import pandas as pd
import re
from pathlib import Path
from shutil import copy, rmtree
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from tqdm import tqdm
from nn_utils import get_video_desc

FILE_COL = 'image'
CLASS_COL = 'level'
CLASS_THRES = 1

def run(input_path, output_path, k, labels_path):
    files = os.listdir(input_path)
    df = pd.read_csv(labels_path)
    df[FILE_COL] = df[FILE_COL].astype(str)
    df[CLASS_COL] = df[CLASS_COL].astype(int)
    df.loc[df['level'] > 3, 'level'] = 3
    # X, y = df[FILE_COL], df[CLASS_COL]
    #splitter = StratifiedShuffleSplit(n_splits=k, test_size=val_size if k == 1 else 1 / k)
    splitter = StratifiedKFold(n_splits=k, shuffle=True)

    for i, split in enumerate(splitter.split(df[FILE_COL], df[CLASS_COL])):
        print(f'Creating fold{i}...')

        df_train = df.iloc[split[0]]
        df_val   = df.iloc[split[1]]
        for s, name in zip([df_train, df_val], ['train', 'val']):
            for row in s.itertuples():
                prefix = 'pos' if row[2] > CLASS_THRES else 'neg'
                [copy(join(input_path, f), join(output_path, f'fold{i}', name, prefix, f)) for f in files if row[1] in f]
            #s['level'] = s.level.astype(int)
            s.to_csv(join(output_path, f'fold{i}', f'labels_{name}.csv'), index=False)
            (refine_dataframe(s, files)).to_csv(join(output_path, f'fold{i}', f'labels_{name}_frames.csv'), index=False)


def refine_dataframe(df, files):
    df_refined = pd.DataFrame(columns=df.columns)
    for row in df.itertuples():
        video_desc = get_video_desc(row[1])
        level = row[2]
        prefix = 'pos' if level > 0 else 'neg'

        matching_files = [f for f in files if get_video_desc(f)['eye_id'] == video_desc['eye_id']]
        for match in matching_files:
            df_refined = df_refined.append({'image': match, 'level': 1 if level > 0 else 0}, ignore_index=True)
    
    return df_refined


if __name__ == '__main__':
    a = argparse.ArgumentParser(description='Split Paxos trainings data into train/val set using stratified shuffling')
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--output", help="absolute path to output folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--folds", "-k", help="Number of folds", type=int, default=10)
    a.add_argument("--force", help="Delete existing folder", action='store_true')
    args = a.parse_args()
    

    if args.force:
        rmtree(args.output)

    os.mkdir(args.output)
    for i in range(args.folds):
        os.mkdir(join(args.output, f'fold{i}'))
        sets = ('train', 'val')
        for s in sets:
            os.mkdir(join(args.output, f'fold{i}', s))
            os.mkdir(join(args.output, f'fold{i}', s, 'pos'))
            os.mkdir(join(args.output, f'fold{i}', s, 'neg'))

    run(args.input, args.output, args.folds, args.labels)
