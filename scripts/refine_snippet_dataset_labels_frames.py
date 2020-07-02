import argparse
import os
from os.path import join
from nn_utils import get_video_desc
import pandas as pd
from tqdm import tqdm

COL_CLASS = 'level'

def run(input_path, labels_path, dataset, mode):
    """
    Takes a CSV file that only holds eye-ids. For every frame corresponding to these frames a new entry with the same DR grading will be created.
    :param input_path: Absolute path to input folder
    :param labels_path: Absolute path to label file that will be refined
    :param dataset: Dataset name (train/val/...)
    :param mode: Determines operation mode (snippets / frames)
    :return:
    """
    df = pd.read_csv(labels_path)
    df[COL_CLASS] = df[COL_CLASS].astype(int)
    df_refined = pd.DataFrame(columns=df.columns)

    files = {'pos': os.listdir(join(input_path, dataset, 'pos')),
             'neg': os.listdir(join(input_path, dataset, 'neg'))}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        video_desc = get_video_desc(row['image'])
        level = row['level']
        prefix = 'pos' if row['level'] > 0 else 'neg'

        matching_files = [f for f in files[prefix] if get_video_desc(f)['eye_id'] == video_desc['eye_id'] and (mode == 'frames' or get_video_desc(f)['snippet_id'] == video_desc['snippet_id'])]
        for file in matching_files:
            df_refined = df_refined.append({'image': file, 'level': 1 if level > 0 else 0}, ignore_index=True)

    df_refined.to_csv(join(input_path, f'labels_{dataset}_frames.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="absolute path to base dataset path")
    a.add_argument("--labels", help="absolute path to CSV label file to be refined")
    a.add_argument("--set", help="which set to refine (train/val/test)", default="train")
    a.add_argument("--mode", help="applied to frames or snippets", default="snippets")

    # a.add_argument("--output", help="absolute path to output folder")
    args = a.parse_args()
    run(args.input, args.labels, args.set, args.mode)
