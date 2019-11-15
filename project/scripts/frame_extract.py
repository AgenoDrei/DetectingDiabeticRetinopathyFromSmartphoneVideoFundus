import os
import argparse
import cv2
from joblib import Parallel, delayed
import multiprocessing


def run(image_path, output_path, time_between_frames):
    paths = [f for f in os.listdir(image_path)]
    print(f'EXTRACT> Found {len(paths)} videos in folder {image_path}: {paths}')

    num_cpus = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cpus)(delayed(extract_images)(os.path.join(image_path, p), output_path, p, time_between_frames=1000 // time_between_frames) for p in paths)
    return output_path


def extract_images(image_path, output_path, name, time_between_frames=1000):
    assert os.path.exists(image_path)
    print(f'EXTRACT> Extracting frames from {image_path}')
    count = 0
    vidcap = cv2.VideoCapture(image_path)
    success, image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count * time_between_frames))    # added this line
      success,image = vidcap.read()
      frame_path = os.path.join(output_path, name[:-4] + f'_{count}.jpg')
      #print (f'Writing frame to: {frame_path}')
      cv2.imwrite(frame_path, image)
      count = count + 1
    return output_path


if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to images")
    a.add_argument("--fps", help="how many frames are extracted per second", type=int, edefault=1)
    args = a.parse_args()
    print('EXTRACT> ', args)

    run(args.input, args.output, args.fps)

    # paths = [f for f in os.listdir(args.input)]
    # print(f'UTIL> Found {len(paths)} videos in folder {args.input}: {paths}')
    #
    # num_cpus = multiprocessing.cpu_count()
    # results = Parallel(n_jobs=num_cpus)(delayed(extract_images)(os.path.join(args.input, p), args.output, p, time_between_frames=1000//args.fps) for p in paths)  # n_jobs = number of processes


    #for p in paths:
    #    video_path = os.path.join(args.input, p)
    #    extract_images(video_path, args.output, p)



