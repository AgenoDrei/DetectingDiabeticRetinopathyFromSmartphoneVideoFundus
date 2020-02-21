import argparse
import re
import os
from pathlib import Path
import pandas as pd
from os.path import join
from nn_processing import copy_corresponding_files
from nn_utils import get_video_desc
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from time_wrap import profile


@profile
def run(input_path, output_path, label_path, val_size=0.2, file_type='.png'):
    assert not os.path.exists(output_path), 'Output folder cannot exist!'
    create_folder_structure(output_path)
    new_labels = move_files_into_class_folders(input_path, output_path, label_path, file_type)
    label_train, label_val = split_data_into_sets(output_path, new_labels, val_size)
    expand_df_with_snippet_info(label_train, label_val, output_path)


def create_folder_structure(path):
    os.mkdir(path)
    os.mkdir(join(path, 'pos'))
    os.mkdir(join(path, 'neg'))
    os.mkdir(join(path, 'train'))
    os.mkdir(join(path, 'val'))
    os.mkdir(join(path, 'train', 'pos'))
    os.mkdir(join(path, 'train', 'neg'))
    os.mkdir(join(path, 'val', 'pos'))
    os.mkdir(join(path, 'val', 'neg'))


def move_files_into_class_folders(input_path, output_path, label_path, file_type, df_name='labels_paxos.csv'):
    name_pattern = re.compile(r"([A-Z])(\d){3}[RL](\d)?")
    all_files = list(Path(input_path).rglob(f'*{file_type}'))
    filtered_files = [str(f.absolute()) for f in all_files if name_pattern.search(str(f)) is not None]

    df = pd.read_excel(label_path, sheet_name='Paxos 4.7')
    processed_df = pd.DataFrame(columns=['image', 'level'])
    df = df[['Eye_ID', 'Pat_ID', 'Paxos_DR', 'Paxos_sharpness']]

    excluded = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Class selection'):
        if isinstance(row['Paxos_DR'], str) or isinstance(row['Paxos_sharpness'], str) or row['Paxos_sharpness'] < 3:
            excluded.append((row["Eye_ID"], row["Paxos_sharpness"], row["Paxos_DR"]))
            # print(f'Skipping row ({row["Eye_ID"]}): Sharpness {row["Paxos_sharpness"]}, Level {row["Paxos_DR"]}')
            continue
        if row['Paxos_DR'] <= 1 or row['Paxos_DR'] == 9:
            processed_df = processed_df.append({'image': row.Eye_ID, 'level': 0}, ignore_index=True)
            copy_corresponding_files(row.Eye_ID, filtered_files, output_path, "neg")
        elif 1 < row['Paxos_DR'] < 5:
            processed_df = processed_df.append({'image': row.Eye_ID, 'level': row['Paxos_DR']}, ignore_index=True)
            copy_corresponding_files(row.Eye_ID, filtered_files, output_path, "pos")

    processed_df.to_csv(join(output_path, df_name), index=False)
    print('Because of missing information or low quality, these eye videos were excluded: ')
    [print(f'{exc[0]}: Lvl {exc[2]} (Reason: {exc[1]})') for exc in excluded]
    return df_name


def split_data_into_sets(output_path, label_path, val_size, df_name=('labels_train.csv', 'labels_val.csv')):
    df = pd.read_csv(join(output_path, label_path))
    df['image'] = df.image.astype(str)
    df_val = pd.DataFrame(columns=df.columns)
    df_train = pd.DataFrame(columns=df.columns)

    x, y = df['image'], df['level']
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
    split = next(splitter.split(x, y))

    for idx in tqdm(split[0], total=len(split[0]), desc='Train data'):  # idx of train set
        df_train = move_set(df.iloc[idx, 0], df.iloc[idx, 1], df_train, output_path, 'train')
    for idx in tqdm(split[1], total=len(split[1]), desc='Val data'):  # idx of validation set
        df_val = move_set(df.iloc[idx, 0], df.iloc[idx, 1], df_val, output_path, 'val')

    df_train.to_csv(join(output_path, df_name[0]), index=False)
    df_val.to_csv(join(output_path, df_name[1]), index=False)

    return df_name


def expand_df_with_snippet_info(label_train, label_val, output_path):
    for set_str in ['train', 'val']:
        df = pd.read_csv(join(output_path, label_train)) if set_str == 'train' else pd.read_csv(join(output_path, label_val))
        df_refined = pd.DataFrame(columns=df.columns)
        files = {'pos': os.listdir(join(output_path, set_str, 'pos')),
                 'neg': os.listdir(join(output_path, set_str, 'neg'))}
        splinters = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc='Refining labels'):
            cur_eye_id = row['image']
            prefix = 'pos' if row['level'] == 1 else 'neg'

            corres_files = sorted([f for f in files[prefix] if cur_eye_id in f], reverse=True, key=lambda name: get_video_desc(name)['snippet_id'])
            splintered_video = check_splinter(corres_files)
            if splintered_video:
                splinters.append(cur_eye_id)
                corres_files_splinter = [f for f in corres_files if len(get_video_desc(f)['eye_id']) > 5]
                df_refined = blow_up_df(corres_files_splinter, df_refined,
                                        f'{cur_eye_id}{get_video_desc(corres_files_splinter[0])["eye_id"][-1]}', row['level'])
            df_refined = blow_up_df(corres_files, df_refined, cur_eye_id, row['level'])
        df_refined.to_csv(join(output_path, f'labels_{set_str}_refined.csv'), index=False)
        print('These splintered videos were detected and included in the expanded labels.csv:')
        [print('Splinter: ', spi) for spi in splinters]

def move_set(image_id, level, set_df, path, set_str):
    severity = 1 if level > 0 else 0
    prefix = 'pos' if severity == 1 else 'neg'
    file_list = [join(path, prefix, f) for f in os.listdir(join(path, prefix))]
    set_df = set_df.append({'image': image_id, 'level': severity}, ignore_index=True)
    copy_corresponding_files(image_id, file_list, path, prefix, set_str=set_str, copy_mode=False)
    return set_df


def check_splinter(file_list):
    for file in file_list:
        if len(get_video_desc(file)['eye_id']) > 5:
            # print('Detected splinter for: ', file)
            return True


def blow_up_df(file_list, df, eye_id, eye_level):
    files_ids = set([get_video_desc(f)['snippet_id'] for f in file_list])
    for i, _ in enumerate(files_ids):
        df = df.append({'image': f'{eye_id}_{i:02d}', 'level': eye_level}, ignore_index=True)

    return df


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", "-i", help="absolute path to input folder")
    a.add_argument("--output", "-o", help="absolute path to output folder")
    a.add_argument("--labels", "-l", help="absolute path to input folder")
    a.add_argument("--val_size", "-s", help="Size of the validation set", type=float, default=0.2)
    a.add_argument("--type", help="File type of original files .MOV/.png/.jpg", default=".png")
    args = a.parse_args()
    print(args)

    run(args.input, args.output, args.labels, val_size=args.val_size, file_type=args.type)
