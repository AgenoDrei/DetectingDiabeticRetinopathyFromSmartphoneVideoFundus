import os
import cv2
import argparse
import pandas as pd


WINDOW_NAME = 'main'
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
OUTPUT_NAME = 'annotated_frames.csv'


def run(input_path, labels_path, mode=None):
    df_labels: pd.DataFrame = pd.read_csv(input_path)
    df_labels = df_labels.loc[df_labels.loc['level' != 0]]
    change_list = []
    
    print(f'INFO> Manual annotation of {len(df_labels)} frames starting...')

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    for row in df_labels.itertuples():
        img = cv2.imread(os.path.join(input_path, row.image))
        cv2.imshow(WINDOW_NAME, img)

        key_code = cv2.waitKey(0)
        print(key_code)

        new_class = 0
        if key_code == 110:         # key-code 110 => n
            new_class = 0
        elif key_code == 100:       # key-code 100 => d
            new_class = 1

        if new_class != row.level:
            df_labels.loc[row.Index, 'level'] = new_class
            change_list.append((row.image, row.level, new_class))

    cv2.destroyAllWindows()
    df_labels.to_csv(os.path.join(input_path, OUTPUT_NAME))
    print(f'INFO> Manual annotation finished. {len(change_list)} rows affected.')
    print(change_list)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="absolute path to input folder")
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--file_type", help="Flag for image oder video mode")        # Video functions come later

    args = a.parse_args()
    run(args.input, args.labels, 'frames')