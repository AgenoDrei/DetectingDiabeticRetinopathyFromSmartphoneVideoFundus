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


def run(input_path, output_path, labels_path):
    """
    Split paxos dataset into positive/negative videos and create simpler CSV file usable for training
    :param input_path: Absolute path to paxos adapter video files
    :param output_path: Absolute path to output folder
    :param labels_path: Absolute path for the excel file describing patients
    :return:
    """
    name_pattern = re.compile(r"([A-Z])(\d){3}[RL](\d)?")
    all_files = list(Path(input_path).rglob('*.MOV'))
    filtered_files = [str(f.absolute()) for f in all_files if name_pattern.search(str(f)) is not None]

    df = pd.read_excel(labels_path, sheet_name='Paxos 4.7')
    processed_df = pd.DataFrame(columns=['image', 'level'])
    df = df[['Eye_ID', 'Pat_ID', 'Paxos_DR', 'Paxos_sharpness']]

    for i, row in df.iterrows():
        if isinstance(row['Paxos_DR'], str) or isinstance(row['Paxos_sharpness'], str) or row['Paxos_sharpness'] < 3:
            continue
        if row['Paxos_DR'] <= 1 or row['Paxos_DR'] == 9:
            processed_df = processed_df.append({'image': row.Eye_ID, 'level': 0}, ignore_index=True)
            move_corresponding_files(row.Eye_ID, filtered_files, output_path, "neg")
        elif 2 <= row['Paxos_DR'] <= 4:
            processed_df = processed_df.append({'image': row.Eye_ID, 'level': 1}, ignore_index=True)
            move_corresponding_files(row.Eye_ID, filtered_files, output_path, "pos")

    processed_df.to_csv(join(output_path, 'processed_labels_paxos.csv'))


def move_corresponding_files(id, filtered_files, path, class_id):
    for file in filtered_files:
        if id in file:
            copy(file, join(path, class_id, os.path.basename(file)))
        else:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--output", help="absolute path to output folder")
    args = a.parse_args()

    assert os.path.exists(args.input)
    if os.path.exists(args.output):
        os.rmdir(join(args.output, 'pos'))
        os.rmdir(join(args.output, 'neg'))
        os.rmdir(args.output)

    os.mkdir(args.output)
    os.mkdir(join(args.output, 'pos'))
    os.mkdir(join(args.output, 'neg'))

    run(args.input, args.output, args.labels)





