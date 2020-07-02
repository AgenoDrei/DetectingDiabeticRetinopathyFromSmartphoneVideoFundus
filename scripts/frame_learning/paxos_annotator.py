import os
import sys

import cv2
import argparse
import pandas as pd
from tqdm import tqdm

WINDOW_NAME = 'main'
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
OUTPUT_NAME = 'annotated_frames.csv'
KEYCODES = {'disease': 100, 'non-disease': 110, 'stop-program': 120}


def run(input_path, labels_path, mode=None):
    df_labels: pd.DataFrame = pd.read_csv(labels_path)
    df_labels = df_labels.loc[df_labels['level'] != 0]
    change_list = []
    
    print(f'INFO> Manual annotation of {len(df_labels)} frames starting...')

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.createTrackbar('progress', WINDOW_NAME, 0, 100, lambda a: None)

    for i, row in tqdm(enumerate(df_labels.itertuples()), smoothing=0, total=len(df_labels)):
        prefix = 'pos' if row.level > 0 else 'neg'
        img = cv2.imread(os.path.join(input_path, prefix, row.image))
        cv2.imshow(WINDOW_NAME, img)
        cv2.setWindowTitle(WINDOW_NAME, f'Name: {os.path.splitext(row.image)[0]} - Grade: {row.level}')

        key_code = cv2.waitKey(0)
        while key_code not in KEYCODES.values():
            key_code = cv2.waitKey(0)

        user_input = get_class_from_key(key_code)
        if user_input != row.level:
            df_labels.loc[row.Index, 'level'] = user_input
            change_list.append((row.image, row.level, user_input))

        cv2.setTrackbarPos('progress', WINDOW_NAME, int(i // (len(df_labels) / 100)))

    cv2.destroyAllWindows()
    df_labels.to_csv(os.path.join(input_path, OUTPUT_NAME))
    print(f'INFO> Manual annotation finished. {len(change_list)} rows affected.')
    print(change_list)


def get_class_from_key(code):
    new_class = -1
    if code == KEYCODES['non-disease']:
        new_class = 0
    elif code == KEYCODES['disease']:
        new_class = 1
    elif code == KEYCODES['stop-program']:
        code2 = cv2.waitKey(0)
        if code2 == KEYCODES['stop-program']:
            sys.exit(0)
    return new_class


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--file_type", help="Flag for image oder video mode")        # Video functions come later
    args = a.parse_args()
    print('INFO> ', args)

    run(args.input, args.labels, 'frames')