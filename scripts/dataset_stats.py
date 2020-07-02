import utils
import nn_utils
import pandas as pd
import numpy as np
import argparse


def run(input_path):
    """
    Create statistics for dataset
    :param input_path: Absolute path to the label file of a dataset
    :return:
    """
    df = pd.read_csv(input_path)
    names = {}
    for row in df.itertuples():
        eye_id = nn_utils.get_video_desc(row.image)['eye_id']
        entry = names.get(eye_id)
        if entry:
            names[eye_id] += 1
        else:
            names[eye_id] = 1

    name_arr = np.array([*names.values()])
    print(f'Dataset stats:\n Mean> {name_arr.mean()},\n Standard Deviaton> {name_arr.std()},\n Median> {np.median(name_arr)},\n Histogram(5)> {np.histogram(name_arr, bins=5)}, \n Histogram(10)> {np.histogram(name_arr, bins=10)}')


a = argparse.ArgumentParser()
a.add_argument("--input", "-i", help="Absolute input path")
args = a.parse_args()
print(args)

run(args.input)
