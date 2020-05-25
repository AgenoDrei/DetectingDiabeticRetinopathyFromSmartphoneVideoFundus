import os
import sys
import time
from torch_lr_finder import LRFinder
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.augmentations.transforms import RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from nn_utils import RetinaDataset, save_batch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import pretrainedmodels as ptm


BASE_PATH = '/home/simon/infcuda2/Data'
GPU_ID = 'cuda'
BATCH_SIZE = 1

def run():
    device = torch.device(GPU_ID if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    hyperparameter = {
        'learning_rate': [1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-7],    # 1e-4
        'weight_decay': [0, 1e-3, 5e-4, 1e-4, 1e-5],              # 1e-4
        'num_epochs': 70,                                   # 100
        'weights': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],          # 0.6
        'optimizer': [optim.Adam, optim.SGD],               # Adam
        'image_size': 300,
        'crop_size': 299
    }
    loaders = prepare_dataset('retina', hyperparameter)

    #model: nn.Module = models.resnet50(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 2)
    model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
    num_ft = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ft, 2)

    children = model.features.children()
    for i, child in enumerate(children):
        if i < 0.0 * len(list(children)):
            for param in child.parameters():
                param.require_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-7, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    lr_finder = LRFinder(model, optimizer_ft, criterion, device=device)
    lr_finder.range_test(loaders[0], end_lr=0.1, num_iter=100, step_mode='exp')
    lr_finder.plot()
    lr_finder.reset()
    return 0


def prepare_dataset(base_name: str, hp):
    aug_pipeline_train = A.Compose([
            A.Resize(hp['image_size'], hp['image_size'], always_apply=True, p=1.0),
            A.RandomCrop(hp['crop_size'], hp['crop_size'], always_apply=True, p=1.0),
            #A.RandomSizedCrop((299, 450), 299, 299, p=1.0),
            A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
            #A.CoarseDropout(min_holes=1, max_holes=4, max_width=100, max_height=100, min_width=25, min_height=25, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.0, rotate_limit=25, border_mode=cv2.BORDER_REFLECT, p=0.5),
            #A.IAAPerspective(scale=(0.02, 0.05), p=0.3),
            A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5)], p=0.5),
            A.OneOf([RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2), A.RandomGamma()], p=0.5),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(always_apply=True, p=1.0)
        ], p=1.0)

    aug_pipeline_val = A.Compose([
        A.Resize(hp['image_size'], hp['image_size'], always_apply=True, p=1.0),
        A.CenterCrop(hp['crop_size'], hp['crop_size'], always_apply=True, p=1.0),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)

    train_dataset = RetinaDataset(os.path.join(BASE_PATH, f'{base_name}_labels_train.csv'), os.path.join(BASE_PATH, f'{base_name}_processed_data_train'), augmentations=aug_pipeline_train, file_type='.jpg', balance_ratio=0.25)
    val_dataset = RetinaDataset(os.path.join(BASE_PATH, f'{base_name}_labels_val.csv'), os.path.join(BASE_PATH, f'{base_name}_processed_data_val'), augmentations=aug_pipeline_val, file_type='.jpg')

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    print(f'Dataset info:\n Train size: {len(train_dataset)},\n Validation size: {len(val_dataset)}')
    return train_loader, val_loader


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    run()
    sys.exit(0)
