import argparse
from pathlib import Path
import re
import detect_retina_snippets as drs
import detect_retina_video_frames as drf
import shutil
import os

WORKING_PATH = '/tmp/dr'
RESULTS_PATH = os.path.join(WORKING_PATH, 'results')

def run(input, output, pipeline, fps, majority, mode='snippet'):
    """
    Convert retionpathy videos to usable snippets (2 secs, 20 frames) with <majority> percent of usable frames
    :param input: Absolute path to video folder, all files with .MOV ending will be recursively processed together
    :param output: Absolute path to output folder
    :param pipeline: Absolute path to pretrained SVM pipeline with scaling / preprocessing
    :param fps: Extracted frames per video
    :param majority: Percentage of usable frames, everything above this threshold will be saved to a new snippet
    :param mode: Determines whether video snippets, snippet frames or frames are extracted (snippet, snippet_frames, frames)
    :return:
    """
    name_pattern = re.compile(r"([A-Z])(\d){3}[RL](\d)?")
    if not os.path.exists(output):
        os.mkdir(output)

    for filename in Path(input).rglob('*.MOV'):
        if name_pattern.search(str(filename)) is None:
            continue
        if check_video_existance(os.listdir(output), os.path.splitext(os.path.basename(str(filename)))[0]):
            continue

        print(f'EXT> Working on file {filename} now!')

        if mode == "snippet" or mode == "snippet_frames":
            drs.run(str(filename), Path(WORKING_PATH), pipeline, fps, majority, only_frames=True if mode == "snippet_frames" else False)
        elif mode == "frames":
            drf.run(filename, WORKING_PATH, pipeline)

        result_files = os.listdir(RESULTS_PATH)
        for file in result_files:
            shutil.copy(os.path.join(RESULTS_PATH, file), output)


def check_video_existance(existing_files: list, name: str) -> bool:
    for file in existing_files:
        if name in file:
            print(f'EXT> File {name} was already processed. Skipping...')
            return True
    return False


if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video files")
    a.add_argument("--output", help="path to snippet results")
    a.add_argument("--pipeline", help="path to SVM pretrained model")
    a.add_argument("--fps", help="Number of frames extracted per second", default=10, type=int)
    a.add_argument("--majority", help="Percentage of frames that have to register as retina", default=0.65, type=float)
    a.add_argument("--mode", help="Set mode: snippet/snippet_frames/frames", default='snippet', type=str)

    args = a.parse_args()
    print('EXT> ', args)

    run(args.input, args.output, args.pipeline, fps=args.fps, majority=args.majority, mode=args.mode)
