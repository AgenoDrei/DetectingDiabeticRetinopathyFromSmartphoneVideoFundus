import argparse
import utils as utl
import os
import joblib as job


def run(input_path, output_path):
    files = os.listdir(input_path)
    job.Parallel(n_jobs=-1, verbose=1)(job.delayed(utl.extract_video_frames)(os.path.join(input_path, f), output_path, frames_per_second=10) for f in files)


if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video files")
    a.add_argument("--output", help="path to extracted frames")
    args = a.parse_args()
    print('S2F> ', args)

    assert os.path.exists(args.input)
    if os.path.exists(args.output):
        os.rmdir(args.output)
    os.mkdir(args.output)

    run(args.input, args.output)