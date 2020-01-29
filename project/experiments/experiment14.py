import os
import sys
import time
from os.path import join
from typing import Tuple

import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.augmentations.transforms import RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from nn_utils import RetinaDataset, SnippetDataset, RetinaNet, dfs_freeze, calc_scores_from_confusion_matrix
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import pretrainedmodels as ptm
import argparse

from tqdm import tqdm


def run(base_path, model_path, gpu_name, batch_size, num_epochs):
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    hyperparameter = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optim.Adam,
        'image_size': 320,
        'crop_size': 299,
        'freeze': 0.0,
        'stump_pooling': False,
        'balance': 0.5
    }
    hyperparameter_str = str(hyperparameter).replace(', \'', ',\n \'')[1:-1]
    print(f'Hyperparameter info:\n {hyperparameter_str}')
    loaders = prepare_dataset(os.path.join(base_path, ''), hyperparameter)

    net:RetinaNet = prepare_model(model_path, hyperparameter)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=hyperparameter['learning_rate'], weight_decay=hyperparameter['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.3, patience=5, verbose=True)

    writer = SummaryWriter(comment=f"video_allsnippets_{net.__class__.__name__}_{hyperparameter['crop_size']}^2_{optimizer_ft.__class__.__name__}")
    model = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device, writer, num_epochs=hyperparameter['num_epochs'])


def prepare_model(model_path, hp):
    stump:ptm.inceptionv4 = ptm.inceptionv4()

    num_ftrs = stump.last_linear.in_features
    stump.last_linear = nn.Linear(num_ftrs, 2)
    stump.load_state_dict(torch.load(model_path))
    stump.train()

    print('Loaded stump: ', len(stump.features))

    for i, child in enumerate(stump.features.children()):
        if i < len(stump.features) * hp['freeze']:
            for param in child.parameters():
                param.requires_grad = False
            dfs_freeze(child)

    net = RetinaNet(frame_stump=stump, do_avg_pooling=hp['stump_pooling'])
    return net


def prepare_dataset(base_name: str, hp):
    aug_pipeline_train = A.Compose([
            A.Resize(hp['image_size'], hp['image_size'], always_apply=True, p=1.0),
            A.RandomCrop(hp['crop_size'], hp['crop_size'], always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(min_holes=1, max_holes=4, max_width=100, max_height=100, min_width=25, min_height=25, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=90, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.OneOf([A.GaussNoise(p=0.5), A.ISONoise(p=0.5), A.IAAAdditiveGaussianNoise(p=0.25), A.MultiplicativeNoise(p=0.25)], p=0.3),
            A.OneOf([A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), A.GridDistortion(p=0.5)], p=0.3),
            A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5)], p=0.3),
            A.OneOf([RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2), A.RandomGamma()], p=0.3),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(always_apply=True, p=1.0)
        ], p=1.0)

    aug_pipeline_val = A.Compose([
        A.Resize(hp['image_size'], hp['image_size'], always_apply=True, p=1.0),
        A.CenterCrop(hp['crop_size'], hp['crop_size'], always_apply=True, p=1.0),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)

    train_dataset = SnippetDataset(join(base_name, 'labels_train_refined.csv'), join(base_name, 'train'), augmentations=aug_pipeline_train, balance_ratio=hp['balance'])
    val_dataset = SnippetDataset(join(base_name, 'labels_val_refined.csv'), join(base_name, 'val'), augmentations=aug_pipeline_val)

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=False, sampler=sampler, num_workers=hp['batch_size'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=hp['batch_size'])
    print(f'Dataset info:\n Train size: {len(train_dataset)},\n Validation size: {len(val_dataset)}')
    return train_loader, val_loader


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, num_epochs=50):
    since = time.time()
    model.to(device)

    for epoch in range(num_epochs):
        print(f'{time.strftime("%H:%M:%S")}> Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        cm = torch.zeros(2, 2)

        # Iterate over data.
        for i, batch in tqdm(enumerate(loaders[0]), total=len(loaders[0]), desc=f'Epoch {epoch}'):
            inputs = batch['frames'].to(device, dtype=torch.float)
            labels = batch['label'].to(device)
            
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            for true, pred in zip(labels, pred):
                cm[int(true), int(pred)] += 1

        train_scores = calc_scores_from_confusion_matrix(cm)
        print(f'Training scores:\n F1: {train_scores["f1"]},\n Precision: {train_scores["precision"]},\n Recall: {train_scores["recall"]}')
        writer.add_scalar('train/f1', train_scores['f1'], epoch)
        writer.add_scalar('train/loss', running_loss / len(loaders[0].dataset), epoch)
        val_loss, _ = validate(model, criterion, loaders[1], device, writer, epoch)

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    torch.save(model.state_dict(), f'model_{time.strftime("%Y%m%d_%H%M")}')
    return model


def validate(model, criterion, loader, device, writer, cur_epoch) -> Tuple[float, float]:
    model.eval()
    cm = torch.zeros(2, 2)
    running_loss = 0.0

    for i, batch in enumerate(loader):
        inputs = batch['frames'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)

        for true, pred in zip(labels, preds):
            cm[true, pred] += 1

    scores = calc_scores_from_confusion_matrix(cm)
    writer.add_scalar('test/f1', scores['f1'], cur_epoch)
    writer.add_scalar('test/precision', scores['precision'], cur_epoch)
    writer.add_scalar('test/recall', scores['recall'], cur_epoch)
    writer.add_scalar('test/loss', running_loss / len(loader.dataset), cur_epoch)
    print(f'Validation scores:\n F1: {scores["f1"]},\n Precision: {scores["precision"]},\n Recall: {scores["recall"]}')

    return running_loss / len(loader.dataset), scores['f1']


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser(description='Train your eyes out')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--model', help='Path for the base model', type=str)
    parser.add_argument('--gpu', help='GPU name', type=str, default='cuda:0')
    parser.add_argument('--bs', help='Batch size for training', type=int, default=8)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    args = parser.parse_args()

    run(args.data, args.model, args.gpu, args.bs, args.epochs)
    sys.exit(0)
