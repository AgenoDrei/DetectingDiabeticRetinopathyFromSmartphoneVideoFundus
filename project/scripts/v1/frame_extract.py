import os
import argparse
import re
from pathlib import Path
from joblib import Parallel, delayed
import utils as utl


def run(input_path, output_path, fps):
    all_files = list(Path(input_path).rglob('*.MOV'))
    name_pattern = re.compile(r"([A-Z])(\d){3}[RL](\d)?")
    paths = [str(f.absolute()) for f in all_files if name_pattern.search(str(f)) is not None]

    print(f'EXTRACT> Found {len(paths)} videos in folder {input_path}: {paths}')

    Parallel(n_jobs=-1)(delayed(utl.extract_video_frames)(p, output_path, frames_per_second=fps) for p in paths)
    return output_path



if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to images")
    a.add_argument("--fps", help="how many frames are extracted per second", type=int, default=1)
    args = a.parse_args()
    print('EXTRACT> ', args)

    os.mkdir(args.output)

    run(args.input, args.output, args.fps)



