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
    def __init__(self, csv_file, root_dir, file_type='.png', balance_ratio=1.0, transform=None, augmentations=None):
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

        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0] + self.file_type)
        img = cv2.imread(img_name)
        #image = image[:,:,[2, 1, 0]]

        severity = self.labels_df.iloc[idx, 1]

        sample = {'image': img, 'label': severity}
        if self.transform:
            sample['image'] = img[:, :, [2, 1, 0]]
            sample['image'] = self.transform(sample['image'])
        if self.augs:
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

        video_name = self.labels_df.iloc[idx, 0]         #TODO: Remove unecassary conversion
        video_index = int(video_name[-2:])
        video_name = video_name[:-2]
        frame_index = set([int(f.split('_')[1]) for f in os.listdir(os.path.join(self.root_dir, prefix)) if video_name in f])
        frame_index = sorted(list(frame_index))
        frame_names = sorted([f for f in os.listdir(os.path.join(self.root_dir, prefix)) if f'{video_name}_{frame_index[video_index]}' in f], key=lambda n: int(n.split('_')[3][:2]))

        sample = {'frames': [], 'label': severity}
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
    def __init__(self, frame_stump):
        super(RetinaNet, self).__init__()
        self.num_frames = 20
        self.out_stump = self.stump.last_linear.in_features
        self.stump = frame_stump
        self.temporal_pooling = nn.MaxPool1d(self.num_frames, stride=1, padding=0, dilation=self.out_stump)

        self.after_pooling = nn.Sequential(nn.Linear(self.out_stump, 256), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, 2))
        #self.fc1 = nn.Linear(self.out_stump, 256)
        #self.fc2 = nn.Linear(256, 2)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        features = []
        for idx in range(0, len(x.size(1))):                        # Iterate over time dimension
            out = self.stump.features(x[:, idx, :, :, :])           # Shove batch trough stump
            out = out.view(out.size(0), -1)                         # Flatten results for fc
            features.append(out)                                    # Size: (B, c*h*w)
        out = torch.cat(features, dim=1)
        out = self.temporal_pooling(out)
        out = self.after_pooling(out)
        return out


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        image = image[top: top + new_h, left: left + new_w]
        return image


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w, _ = image.shape
        #image = trans.resize(image, (h, w))

        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        return image


class RandomNormalCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w, _ = image.shape
        #image = trans.resize(image, (h, w))

        new_h, new_w = self.output_size
        mean_h, mean_w = (h - new_h) // 2, (w - new_w) // 2
        std_h, std_w = mean_h * 0.25, mean_w * 0.25

        top_rand = np.random.normal(mean_h, std_h)
        top = top_rand if top_rand < h - new_h else h - new_h - 1
        top = top if top > 0 else 0
        left_rand = np.random.normal(mean_w, std_w)
        left = left_rand if left_rand < w - new_w else w - new_w - 1
        left = left if left > 0 else 0
        #top = np.random.randint(0, h - new_h)
        #left = np.random.randint(0, w - new_w)
        image = image[int(top): int(top) + new_h, int(left): int(left) + new_w]
        return image


class Flip(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.prob = probability

    def __call__(self, image):
        if np.random.rand() < self.prob:
            image = cv2.flip(image, 1)
        return image


class Blur(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.prob = probability

    def __call__(self, image):
        if np.random.rand() < self.prob:
            image = cv2.GaussianBlur(image, (5, 5) ,0)
        return image


class EnhanceContrast(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.prob = probability

    def __call__(self, image):
        if np.random.rand() < self.prob:
            image = image[:, :, [2, 1, 0]]
            image = utl.enhance_contrast_image(image, clip_limit=np.random.randint(2, 5))
            image = image[:, :, [2, 1, 0]]
        return image


class ToTensor(object):
    def __call__(self, image):
        # swap color axis because, DOES NOT NORMALIZE RIGHT NOW!
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


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


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def calc_scores_from_confusion_matrix(cm):
    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return {'precision': precision, 'recall': recall, 'f1': f1}
