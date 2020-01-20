import argparse
import utils as utl


def run(input_path, output_path):
    utl.extract_video_frames(input_path, output_path, frames_per_second=10)


if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video files")
    a.add_argument("--output", help="path to extracted frames")
    args = a.parse_args()
    print('S2F> ', args)

    run(args.input, args.output)