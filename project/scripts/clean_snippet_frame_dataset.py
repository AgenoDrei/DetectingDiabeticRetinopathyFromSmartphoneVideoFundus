import argparse
import pandas as pd
from nn_utils import get_video_desc
from tqdm import tqdm


def run(labels_path, min_conf=80):
    df = pd.read_csv(labels_path)

    drop_list = []
    for i, row in tqdm(df.iterrows()):
        video_path = row['image']
        video_desc = get_video_desc(video_path)

        if video_desc['confidence'] < min_conf:
            drop_list.append(i)
    df.drop(df.index[drop_list], inplace=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    a = argparse.ArgumentParser()
    a.add_argument("--labels", help="absolute path to input folder")
    a.add_argument("--min_quality", help="minimal svm confidence for snippet frames")
    args = a.parse_args()
    print(args)

    run(args.labels, min_conf=args.min_quality)