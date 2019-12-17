import argparse
from pathlib import Path
import re
import detect_retina_snippets as drs
import shutil
import os

WORKING_PATH = '/tmp/dr'
RESULTS_PATH = os.path.join(WORKING_PATH, 'results')

def run(input, output, pipeline, fps, majority):
    name_pattern = re.compile(r"([A-Z])(\d){3}[RL](\d)?")
    if not os.path.exists(output):
        os.mkdir(output)

    for filename in Path(input).rglob('*.MOV'):
        if name_pattern.search(str(filename)) is None:
            continue
        if check_video_existance(os.listdir(output), os.path.splitext(os.path.basename(str(filename)))[0]):
            continue

        print(f'EXT> Working on file {filename} now!')
        drs.run(str(filename), Path(WORKING_PATH), pipeline, fps, majority)

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
    args = a.parse_args()
    print('EXT> ', args)

    run(args.input, args.output, args.pipeline, fps=args.fps, majority=args.majority)