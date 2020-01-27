import argparse
import os
from os.path import join

import pandas as pd
from tqdm import tqdm


def run(input_path, labels_path, dataset):
    df = pd.read_csv(labels_path)
    df_refined = pd.DataFrame(columns=df.columns)

    criteria = lambda name: int(name.split("_")[1])

    files = {'pos': os.listdir(join(input_path, dataset, 'pos')),
             'neg': os.listdir(join(input_path, dataset, 'neg'))}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        cur_name = row['image']
        prefix = 'pos' if row['level'] == 1 else 'neg'
        corres_files = sorted([f for f in files[prefix] if cur_name in f], reverse=True, key=criteria)
        corres_files_ids = set([int(f.split("_")[1]) for f in corres_files])

        for i, id in enumerate(corres_files_ids):
            df_refined = df_refined.append({'image': f'{cur_name}{i:02d}', 'level': row['level']}, ignore_index=True)

    df_refined.to_csv(join(input_path, f'labels_{dataset}_refined.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--set", help="which set to refine (train/val/test)", default="train")

    # a.add_argument("--output", help="absolute path to output folder")
    args = a.parse_args()
    run(args.input, args.labels, args.set)
