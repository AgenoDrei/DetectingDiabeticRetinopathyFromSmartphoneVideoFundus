import argparse
import os
from os.path import join


def run(input_path):
    """
    Small script to remove splinter from a folder. These get created when a exaimer split an eye examination into two files. E.g. C001L + C001L2.
    All files correspondig to the second video (the splinter) will get ids that are higher than the id of the last frame of the first file
    :param input_path: Absolute path to folder where the splinter files have to be renamed.
    :return:
    """
    files = os.listdir(input_path)
    eyes = [f.split('_')[0] for f in files]
    eyes = set(eyes)
    num_frames = {eye: 0 for eye in eyes}
    file_type = os.path.splitext(files[0])[1]

    for file in files:
        name = file.split('_')[0]
        idx = int(os.path.splitext(file.split('_')[1])[0])
        if idx > num_frames[name]:
            num_frames[name] = idx

    print(f'Found {len([e for e in eyes if len(e) > 5])} splinters in folder')

    for eye in eyes:
        if len(eye) == 5:   # no splinter
            continue
        orig = eye[:5]
        if num_frames.get(orig) is None:
            num_frames[orig] = 0

        print(eye, num_frames[eye])
        splinter_files = sorted([f for f in files if eye in f], key=lambda name: int(os.path.splitext(name.split('_')[1])[0]))
        [os.rename(join(input_path, splinter_files[i]), join(input_path, f'{orig}_{num_frames[orig] + i + 1}{file_type}')) for i in range(len(splinter_files))]


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="absolute path to folder where splinter will be renamed")
    args = a.parse_args()
    print(args)

    run(args.input)
