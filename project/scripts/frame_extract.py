import cv2
import pandas as pd
import sys

OUTPUT_PATH = ''

print(f'Calling {sys.argv[0]} with parameters: {sys.argv}')

df = pd.read_csv(sys.argv[1])

for index, row in df.iterrows():
    video_path = row[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    for i in range(1, len(row)):
        cur_timestamp = row[i]
        vidcap.set(cv2.CAP_PROP_POS_MSEC, cur_timestamp)
        _, frame = vidcap.read()

        cv2.imwrite(f'{video_path[:-3]}{i}png', frame)



