import utils
import nn_utils
import pandas as pd
import numpy as np
import argparse


a = argparse.ArgumentParser()
a.add_argument("--input", "-i", help="Absolute input path")
args = a.parse_args()
print(args)

df = pd.read_csv(args.input) 
names = {}
for row in df.itertuples():
    eye_id = nn_utils.get_video_desc(row.image)['eye_id']
    entry = names.get(eye_id)
    if entry:
        names[eye_id] += 1
    else:
        names[eye_id] = 1

name_arr = np.array([*names.values()])
print(f'Dataset stats:\n {name_arr.mean()},\n {name_arr.std()},\n {np.histogram(name_arr, bins=5)}, \n {np.histogram(name_arr, bins=10)}')
