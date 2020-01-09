import os
import sys
import time

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

BASE_PATH = '/home/user/mueller9/Data'
#BASE_PATH = '/data/simon/'
GPU_ID = 'cuda:5'
BATCH_SIZE = 16

def run():
    writer = SummaryWriter(comment="_exp2_90000_resnet50__672x672_frozen")
    device = torch.device(GPU_ID if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    loaders = prepare_dataset('retina')
    hyperparameter = {
        'learning_rate': [1e-2, 1e-3, 3e-4, 1e-4, 3e-5],    # 1e-4
        'weight_decay': [0, 1e-3, 5e-4, 1e-4],              # 1e-4
        'num_epochs': 50,                                   # 100
        'weights': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],          # 0.6
        'optimizer': [optim.Adam, optim.SGD],                # Adam
        'size': 598,
    }

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    layer_idx = 0
    for child in model.children():
        layer_idx += 1
        if layer_idx < len(list(model.children())) * 0.5:
            for param in child.parameters():
                param.requires_grad = False

    optimizer_ft = optim.Adam(model.parameters(), lr=hyperparameter['learning_rate'][3], weight_decay=hyperparameter['weight_decay'][0])
    criterion = nn.CrossEntropyLoss()
    step_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[15, 25, 35, 45], gamma=0.1)

    save_batch(next(iter(loaders[0])), '/tmp')
    model = train_model(model, criterion, optimizer_ft, step_scheduler, loaders, device, writer, num_epochs=hyperparameter['num_epochs'])

def prepare_dataset(base_name: str):
    aug_pipeline_train = A.Compose([
            A.Resize(900, 900, always_apply=True, p=1.0),
            A.RandomCrop(672, 672, always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(min_holes=1, max_holes=4, max_width=100, max_height=100, min_width=25, min_height=25, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            #A.IAAPerspective(scale=(0.02, 0.05), p=0.5),
            A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5)]),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(p=0.5),
            A.CLAHE(clip_limit=(2, 4), p=1.0),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(always_apply=True, p=1.0)
            #A.OneOf([RandomBrightnessContrast(), A.RandomGamma(), CLAHE(clip_limit=4), Blur(blur_limit=9)], p=0.5),
            #A.OneOf([A.ElasticTransform(alpha=120, sigma=6, alpha_affine=3.6), A.GridDistortion(), A.OpticalDistortion(distort_limit=2, shift_limit=0.5)], p=0.5),
            #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], p=1.0)

    aug_pipeline_val = A.Compose([
        A.Resize(900, 900, always_apply=True, p=1.0),
        A.CenterCrop(672, 672, always_apply=True, p=1.0),
        A.CLAHE(clip_limit=(3, 3), p=1.0),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)

    train_dataset = RetinaDataset(os.path.join(BASE_PATH, f'{base_name}_labels_train.csv'), os.path.join(BASE_PATH, f'{base_name}_data_train'), augmentations=aug_pipeline_train, file_type='.jpg', balance_ratio=0.25)
    val_dataset = RetinaDataset(os.path.join(BASE_PATH, f'{base_name}_labels_val.csv'), os.path.join(BASE_PATH, f'{base_name}_data_val'), augmentations=aug_pipeline_val, file_type='.jpg')

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    print(f'Dataset info:\n Train size: {len(train_dataset)},\n Validation size: {len(val_dataset)}')
    return train_loader, val_loader


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, num_epochs=50):
    since = time.time()
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        print(f'{time.strftime("%H:%M:%S")}> Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        cm = torch.zeros(2, 2)

        # Iterate over data.
        for i, batch in enumerate(loaders[0]):
            inputs = batch['image'].to(device, dtype=torch.float)
            labels = batch['label'].to(device)
            
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            #loss2 = criterion(aux_outputs, labels)
            #loss = loss1 + 0.4 * loss2
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            for true, pred in zip(labels, preds):
                cm[int(true), int(pred)] += 1
            #running_f1 += metrics.f1_score(labels.cpu(), preds.cpu()) * inputs.size(0)

        scheduler.step()
        print(cm)
        writer.add_scalar('train/loss', running_loss / len(loaders[0].dataset), epoch)
        validate(model, criterion, loaders[1], device, writer, epoch)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #print(f'Best val Acc: {best_acc:4f}')

    validate(model, criterion, loaders[1], device, writer, num_epochs)

    torch.save(model.state_dict(), os.path.join(BASE_PATH, f'model{time.time()}.dat'))
    return model


def validate(model, criterion, loader, device, writer, cur_epoch):
    model.eval()
    cm = torch.zeros(2, 2)
    running_loss = 0.0

    for i, batch in enumerate(loader):
        inputs = batch['image'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)

        for true, pred in zip(labels, preds):
            cm[true, pred] += 1

    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()

    precision = tp / (tp + fp + 0.1)
    recall = tp / (tp + fn + 0.1)
    f1 = (2 * precision * recall) / (precision + recall + 0.1)

    writer.add_scalar('test/f1', f1, cur_epoch)
    writer.add_scalar('test/precision', precision, cur_epoch)
    writer.add_scalar('test/recall', recall, cur_epoch)
    writer.add_scalar('test/loss', running_loss / len(loader.dataset), cur_epoch)
    print(cm)
    print(f'Scores:\n F1: {f1},\n Precision: {precision},\n Recall: {recall}')


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    run()
    sys.exit(0)
