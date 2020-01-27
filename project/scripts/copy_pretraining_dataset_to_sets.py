import pandas as pd
import os
from os import path
from sklearn.utils import shuffle
from tqdm import tqdm

BASE_PATH = "/data/simon/Datasets/2015-2019-blindness-detection-images/"
LABELS_PATH = "retina_labels.csv"
DATA_PATH = "retina_data"
TEST_PATH, VAL_PATH = "retina_data_test", "retina_data_val"

os.mkdir(path.join(BASE_PATH, TEST_PATH))
os.mkdir(path.join(BASE_PATH, VAL_PATH))

df:pd.DataFrame = pd.read_csv(path.join(BASE_PATH, LABELS_PATH))
df = shuffle(df)

df['image'] = df.image.astype(str)
df['level'] = df.level.map(lambda v: 0 if v <= 1 else 1)
print(df.level.value_counts())

df_val = pd.DataFrame(columns=df.columns)
df_test = pd.DataFrame(columns=df.columns)
df_train = pd.DataFrame(columns=df.columns)
df.reset_index(inplace=True, drop=True)

for i, row in tqdm(df.iterrows(), total=len(df)):
    if i < 0.05 * len(df):
        df_val = df_val.append(row.to_dict(), ignore_index=True)
        os.rename(path.join(BASE_PATH, DATA_PATH, f'{row.image}.jpg'), path.join(BASE_PATH, VAL_PATH, f'{row.image}.jpg'))
        continue
    elif 0.05 * len(df) <= i < 0.1 * len(df):
        df_test = df_test.append(row.to_dict(), ignore_index=True)
        os.rename(path.join(BASE_PATH, DATA_PATH, f'{row.image}.jpg'), path.join(BASE_PATH, TEST_PATH, f'{row.image}.jpg'))
        continue
    else:
        df_train = df_train.append(row.to_dict(), ignore_index=True)
        continue

print(f'Len train: {len(df_train)}, Len val: {len(df_val)}, Len test: {len(df_test)}')
df_train.to_csv(path.join(BASE_PATH, 'retina_labels_train.csv'), index=False)
df_val.to_csv(path.join(BASE_PATH, 'retina_labels_val.csv'), index=False)
df_test.to_csv(path.join(BASE_PATH, 'retina_labels_test.csv'), index=False)
