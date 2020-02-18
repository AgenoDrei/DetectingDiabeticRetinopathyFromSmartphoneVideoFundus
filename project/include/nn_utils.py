import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import Dataset
from torchvision import utils
import utils as utl
import nn_processing


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

        sample = {'image': img, 'label': severity, 'eye_id': get_video_desc(self.labels_df.iloc[idx, 0], only_eye=True)['eye_id']}
        if self.transform:
            sample['image'] = img[:, :, [2, 1, 0]]
            sample['image'] = self.transform(sample['image'])
        if self.augs:
            sample['image'] = self.augs(image=img)['image']
        return sample


class FiveCropRetinaDataset(Dataset):
    def __init__(self, csv_file, root_dir, size, file_type='.png', balance_ratio=1.0, augmentations=None, use_prefix=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
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
        assert not torch.is_tensor(idx)
        severity = self.labels_df.iloc[idx // self.num_crops, 1]
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
            img = cv2.resize(img, (self.size[0], self.size[0]))
            img = utl.do_five_crop(img, self.size[1], self.size[1], crop_idx)
            sample['image'] = self.augs(image=img)['image']
        return sample


class SnippetDataset(Dataset):
    def __init__(self, csv_file, root_dir, file_type='.png', balance_ratio=1.0, validation=False, augmentations=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.file_type = file_type
        self.augs = augmentations
        self.ratio = balance_ratio
        self.mode = 'val' if validation else 'train'
        self.num_crops = 5

    def __len__(self):
        return len(self.labels_df) if self.mode == 'train' else len(self.labels_df) * self.num_crops

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)

        image_idx = idx // self.num_crops if self.mode == 'val' else idx
        crop_idx = idx % self.num_crops

        severity = self.labels_df.iloc[image_idx, 1]
        prefix = 'pos' if severity == 1 else 'neg'

        video_name = self.labels_df.iloc[image_idx, 0]
        video_desc = get_video_desc(video_name)

        files = [f for f in os.listdir(os.path.join(self.root_dir, prefix)) if video_desc['eye_id'] == get_video_desc(f)['eye_id']]
        #if len(frame_index) - 1 < video_index:
        #    print('Problem with video ', video_name, video_index)

        frame_names = sorted([f for f in files if video_desc['snippet_id'] == get_video_desc(f)['snippet_id']], key=lambda n: get_video_desc(n)['frame_id'])
        #print(len(frame_names))

        sample = {'frames': [], 'label': severity, 'name': video_desc['eye_id'][:5]}
        crop_state = np.random.randint(0, 5) if self.mode == 'train' else crop_idx
        for name in frame_names:
            img = cv2.imread(os.path.join(self.root_dir, prefix, name))
            img =  self.augs(image=img)['image'] if self.augs else img                  # TODO: Maybe improve later on
            # Apply cropping after image augmentations, continue with transformation afterwards
            img = utl.do_five_crop(img, 299, 299, state=crop_state)
            img = alb.Normalize().apply(img)
            img = ToTensorV2().apply(img)
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

        self.after_pooling = nn.Sequential(nn.Linear(self.out_stump, self.pool_params[1]), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(self.pool_params[1], 2))
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


def get_video_desc(video_path, only_eye = False):
    """
    Get video description in easy usable dictionary
    :param video_path: path / name of the video_frame file
    :param only_eye: Only returns the first part of the string
    :return: dict(eye_id, snippet_id, frame_id, confidence), only first two are required
    """
    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    info_parts = video_name.split("_")

    if len(info_parts) == 1 or only_eye:
        return {'eye_id': info_parts[0]}
    elif len(info_parts) == 2:
        return {'eye_id': info_parts[0], 'snippet_id': int(info_parts[1])}
    elif len(info_parts) > 3:
        return {'eye_id': info_parts[0], 'snippet_id': int(info_parts[1]), 'frame_id': int(info_parts[3]), 'confidence': info_parts[2]}
    else:
        return {'eye_id': ''}

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def calc_scores_from_confusion_matrix(cm):
    """
    Calc precision, recall and f1 score from a 2x2 numpy confusion matrix
    :param cm: confusion matrix, numpy 2x2 ndarray
    :return: dict(precision, recall, f1)
    """
    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()

    if tp + fp == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0}

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return {'precision': precision, 'recall': recall, 'f1': f1}


class MajorityDict:
    def __init__(self):
        self.dict = {}

    def add(self, predictions, ground_truth, key_list):
        """
        Add network predictions to Majority Dict, has to be called for every batch of the validation set
        :param predictions: list of predictions
        :param ground_truth: list of correct, known labels
        :param key_list: list of keys (like video id)
        :return: None
        """
        for i, (true, pred) in enumerate(zip(ground_truth, predictions)):
            if self.dict.get(key_list[i]):
                entry = self.dict[key_list[i]]
                entry[str(pred)] += 1
                entry['count'] += 1
            else:
                self.dict[key_list[i]] = {'0': 0 if int(pred) else 1, '1': 1 if int(pred) else 0, 'count': 1, 'label': int(true)}

    def get_predictions_and_labels(self, ratio: float = 0.5):
        """
        Do majority voting to get final predicions aggregated over all elements sharing the same key
        :param ratio: Used to shange majority percentage (default 50/50)
        :return: dict(predictions, labels)
        """
        labels, preds = [], []
        for i, item in self.dict.items():
            if item['1'] > ratio * (item['0'] + item['1']):
                preds.append(1)
            else:
                preds.append(0)
            labels.append(item['label'])
        return {'predictions': preds, 'labels': labels}

    def get_roc_data(self, step_size: float = 0.05):
        """
        Generate predictions for different thresholds
        :param step_size: step_size between different thresholds
        :return: dict(step: dict)
        """
        roc_data = {}
        for i in np.arange(0, 1.0, step_size):
            roc_data[i] = self.get_predictions_and_labels(ratio=i)
        return roc_data
