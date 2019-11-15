import os
import argparse
import cv2
from joblib import Parallel, delayed
import multiprocessing

TIME_BETWEEN_FRAMES = 1000

def extract_images(image_path, output_path, name):
    print(f'UTIL> Extracting frames from {image_path}')
    count = 0
    vidcap = cv2.VideoCapture(image_path)
    success, image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*TIME_BETWEEN_FRAMES))    # added this line
      success,image = vidcap.read()
      frame_path = os.path.join(output_path, name[:-4] + f'_{count}.jpg')
      #print (f'Writing frame to: {frame_path}')
      cv2.imwrite(frame_path, image)
      count = count + 1


if __name__== '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="path to video")
    a.add_argument("--output", help="path to images")
    args = a.parse_args()
    print(args)

    paths = [f for f in os.listdir(args.input)]
    print(f'UTIL> Found {len(paths)} videos in folder {args.input}: {paths}')

    num_cpus = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cpus)(delayed(extract_images)(os.path.join(args.input, p), args.output, p) for p in paths)  # n_jobs = number of processes


    #for p in paths:
    #    video_path = os.path.join(args.input, p)
    #    extract_images(video_path, args.output, p)



