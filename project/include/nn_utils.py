import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

from torch import nn
from torch.utils.data import Dataset
from torchvision import utils
from skimage import io
from skimage import transform as trans
import utils as utl


class RetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, file_type='.png', balance_ratio=1.0, transform=None, augmentations=None, use_prefix=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert transform is None or augmentations is None
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.transform = transform
        self.augs = augmentations
        self.ratio = balance_ratio
        self.use_prefix = use_prefix

    def __len__(self):
        return len(self.labels_df)

    def get_weight(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        severity = self.labels_df.iloc[idx, 1]
        return self.ratio if severity == 0 else 1.0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        severity = self.labels_df.iloc[idx, 1]

        if self.use_prefix:
            prefix = 'pos' if severity == 1 else 'neg'
        else:
            prefix = ''
        img_name = os.path.join(self.root_dir, prefix, self.labels_df.iloc[idx, 0] + self.file_type)
        img = cv2.imread(img_name)
        #image = image[:,:,[2, 1, 0]]

        sample = {'image': img, 'label': severity}
        if self.transform:
            sample['image'] = img[:, :, [2, 1, 0]]
            sample['image'] = self.transform(sample['image'])
        if self.augs:
            sample['image'] = self.augs(image=img)['image']
        return sample


class FiveCropRetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, file_type='.png', balance_ratio=1.0, size=299, augmentations=None, use_prefix=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.augs = augmentations
        self.ratio = balance_ratio
        self.use_prefix = use_prefix
        self.num_crops = 5
        self.size = size

    def __len__(self):
        return len(self.labels_df) * self.num_crops

    def get_weight(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        severity = self.labels_df.iloc[idx, 1]
        return self.ratio if severity == 0 else 1.0

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)

        image_idx = idx // self.num_crops
        crop_idx = idx % self.num_crops

        severity = self.labels_df.iloc[image_idx, 1]

        if self.use_prefix:
            prefix = 'pos' if severity == 1 else 'neg'
        else:
            prefix = ''

        img_name = os.path.join(self.root_dir, prefix, self.labels_df.iloc[image_idx, 0] + self.file_type)
        img = cv2.imread(img_name)
        #image = image[:,:,[2, 1, 0]]

        sample = {'image': img, 'label': severity, 'image_idx': image_idx}
        if self.augs:
            img = utl.do_five_crop(img, self.size, self.size, crop_idx)
            sample['image'] = self.augs(image=img)['image']
        return sample


class SnippetDataset(Dataset):
    def __init__(self, csv_file, root_dir, file_type='.png', balance_ratio=1.0, transform=None, augmentations=None):
        assert transform is None or augmentations is None
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.augs = augmentations
        self.ratio = balance_ratio

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        severity = self.labels_df.iloc[idx, 1]
        prefix = 'pos' if severity == 1 else 'neg'

        video_name = self.labels_df.iloc[idx, 0]
        video_desc = get_video_desc(video_name)

        files = [f for f in os.listdir(os.path.join(self.root_dir, prefix)) if video_desc['eye_id'] == get_video_desc(f)['eye_id']]
        #if len(frame_index) - 1 < video_index:
        #    print('Problem with video ', video_name, video_index)

        frame_names = sorted([f for f in files if video_desc['snippet_id'] == get_video_desc(f)['snippet_id']], key=lambda n: get_video_desc(n)['frame_id'])
        #print(len(frame_names))

        sample = {'frames': [], 'label': severity, 'name': video_desc['eye_id'][:5]}
        for name in frame_names:
            img = cv2.imread(os.path.join(self.root_dir, prefix, name))
            img =  self.augs(image=img)['image'] if self.augs else img
            sample['frames'].append(img)

        sample['frames'] = torch.stack(sample['frames']) if self.augs else np.stack(sample['frames'])
        return sample

    def get_weight(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        severity = self.labels_df.iloc[idx, 1]
        return self.ratio if severity == 0 else 1.0


class RetinaNet(nn.Module):
    def __init__(self, frame_stump, do_avg_pooling=True):
        super(RetinaNet, self).__init__()
        self.stump = frame_stump
        self.pool_stump = do_avg_pooling
        self.num_frames = 20
        self.pool_params = (self.stump.last_linear.in_features, 256) if self.pool_stump else (98304, 1024)
        self.out_stump = self.pool_params[0]

        self.avg_pooling = self.stump.avg_pool
        self.temporal_pooling = nn.MaxPool1d(self.num_frames, stride=1, padding=0, dilation=self.out_stump)

        self.after_pooling = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(self.out_stump, self.pool_params[1]), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(self.pool_params[1], 2))
        #self.fc1 = nn.Linear(self.out_stump, 256)
        #self.fc2 = nn.Linear(256, 2)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        features = []
        for idx in range(0, x.size(1)):                             # Iterate over time dimension
            out = self.stump.features(x[:, idx, :, :, :])           # Shove batch trough stump
            out = self.avg_pooling(out) if self.pool_stump else out
            out = out.view(out.size(0), -1)                         # Flatten results for fc
            features.append(out)                                    # Size: (B, c*h*w)
        out = torch.cat(features, dim=1)
        out = self.temporal_pooling(out.unsqueeze(dim=1))
        out = self.after_pooling(out.view(out.size(0), -1))
        return out


def display_examples(ds):
    fig = plt.figure(figsize=(10, 10))

    for i in range(0, 40, 10):
        sample = ds[i]
        ax = plt.subplot(1, 4, i // 10 + 1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i}- {sample["label"]}')
        ax.axis('off')
        plt.imshow(sample['image'])

    plt.show()


# Helper function to show a batch
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, label_batch = sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()


def save_batch(batch, path):
    images_batch, label_batch = batch['image'], batch['label']
    for i, img in enumerate(images_batch):
        cv2.imwrite(os.path.join(path, f'{i}_{label_batch[i]}.png'), img.numpy().transpose((1, 2, 0)))


def get_video_desc(video_path):
    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    info_parts = video_name.split("_")

    if len(info_parts) == 2:
        return {'eye_id': info_parts[0], 'snippet_id': int(info_parts[1])}
    else:
        return {'eye_id': info_parts[0], 'snippet_id': int(info_parts[1]), 'frame_id': int(info_parts[3]), 'confidence': info_parts[2]}


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def calc_scores_from_confusion_matrix(cm):
    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()

    if tp + fp == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0}

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return {'precision': precision, 'recall': recall, 'f1': f1}
