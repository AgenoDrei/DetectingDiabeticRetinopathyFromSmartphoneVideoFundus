import joblib as j
import argparse
import cv2
import os
import numpy as np
from os.path import join
import pandas as pd
import re
from pathlib import Path
from shutil import copy


def run(input_path, output_path, labels_path, ignore_files=False):
    """
    Split paxos dataset into positive/negative videos and create simpler CSV file usable for training
    :param input_path: Absolute path to paxos adapter video files
    :param output_path: Absolute path to output folder
    :param labels_path: Absolute path for the excel file describing patients
    :return:
    """
    name_pattern = re.compile(r"([A-Z])(\d){3}[RL](\d)?")
    all_files = list(Path(input_path).rglob('*.jpg'))
    filtered_files = [str(f.absolute()) for f in all_files if name_pattern.search(str(f)) is not None]

    df = pd.read_excel(labels_path, sheet_name='Paxos 4.7')
    processed_df = pd.DataFrame(columns=['image', 'level'])
    df = df[['Eye_ID', 'Pat_ID', 'Paxos_DR', 'Paxos_sharpness']]

    for i, row in df.iterrows():
        if isinstance(row['Paxos_DR'], str) or isinstance(row['Paxos_sharpness'], str) or row['Paxos_sharpness'] < 3:
            print('Skipping row: ', row['Eye_ID'], row['Paxos_sharpness'], row['Paxos_DR'])
            continue
        if row['Paxos_DR'] <= 1 or row['Paxos_DR'] == 9:
            processed_df = processed_df.append({'image': row.Eye_ID, 'level': 0}, ignore_index=True)
            if not ignore_files: move_corresponding_files(row.Eye_ID, filtered_files, output_path, "neg")
        elif row['Paxos_DR'] > 1 and row['Paxos_DR'] < 5:
            processed_df = processed_df.append({'image': row.Eye_ID, 'level': row['Paxos_DR']}, ignore_index=True)
            if not ignore_files: move_corresponding_files(row.Eye_ID, filtered_files, output_path, "pos")

    processed_df.to_csv(join(output_path, 'processed_labels_paxos.csv'), index=False)


def move_corresponding_files(id, filtered_files, path, class_id):
    for file in filtered_files:
        if id in file:
            copy(file, join(path, class_id, os.path.basename(file)))
        else:
            continue


if __name__ == '__main__':
    a = argparse.ArgumentParser(description='Split paxos dataset into positive/negative videos and create CSV file usable for training')
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--output", help="absolute path to output folder")
    a.add_argument("--ignore_files", help="Do not move video files, just recreate csv", default=False, type=bool)
    args = a.parse_args()
    print(args)

    assert os.path.exists(args.input)
    if os.path.exists(args.output) and not args.ignore_files:
        os.rmdir(join(args.output, 'pos'))
        os.rmdir(join(args.output, 'neg'))
        os.rmdir(args.output)

    if not args.ignore_files:
        os.mkdir(args.output)
        os.mkdir(join(args.output, 'pos'))
        os.mkdir(join(args.output, 'neg'))

    run(args.input, args.output, args.labels, ignore_files=args.ignore_files)





