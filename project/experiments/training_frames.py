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
from multichannel_inceptionv4 import my_inceptionv4
from narrow_inceptionv import NarrowInceptionV1
from nn_processing import ThresholdGlare
from nn_utils import RetinaDataset, SnippetDataset, RetinaNet, dfs_freeze, calc_scores_from_confusion_matrix, get_video_desc, MajorityDict, \
    MultiChannelRetinaDataset, write_scores, write_f1_curve, write_pr_curve
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import pretrainedmodels as ptm
import argparse
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm


def run(base_path, model_path, gpu_name, batch_size, num_epochs):
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    hyperparameter = {
        'data': os.path.basename(os.path.normpath(base_path)),
        'learning_rate': 1e-4,
        'weight_decay': 3e-4,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optim.Adam.__name__,
        'freeze': 0.0,
        'balance': 0.45,
        'image_size': 450,
        'crop_size': 399,
        'pretraining': True,
        'preprocessing': False,
        'multi_channel': False,
        'boosting': 2.00,
        'use_clahe': False,
        'narrow_model': False,
        'remove_glare': False
    }
    aug_pipeline_train = A.Compose([
        A.CLAHE(always_apply=hyperparameter['use_clahe'], p=1.0 if hyperparameter['use_clahe'] else 0.0),
        ThresholdGlare(always_apply=hyperparameter['remove_glare'], p=1.0 if hyperparameter['remove_glare'] else 0.0),
        A.Resize(hyperparameter['image_size'], hyperparameter['image_size'], always_apply=True, p=1.0),
        A.RandomCrop(hyperparameter['crop_size'], hyperparameter['crop_size'], always_apply=True, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(min_holes=1, max_holes=3, max_width=75, max_height=75, min_width=25, min_height=25, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.OneOf([A.GaussNoise(p=0.5), A.ISONoise(p=0.5), A.IAAAdditiveGaussianNoise(p=0.25), A.MultiplicativeNoise(p=0.25)], p=0.3),
        A.OneOf([A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), A.GridDistortion(p=0.5)], p=0.3),
        A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5)], p=0.3),
        A.OneOf([RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2), A.RandomGamma()], p=0.3),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)
    aug_pipeline_val = A.Compose([
        A.CLAHE(always_apply=hyperparameter['use_clahe'], p=1.0 if hyperparameter['use_clahe'] else 0.0),
        ThresholdGlare(always_apply=hyperparameter['remove_glare'], p=1.0 if hyperparameter['remove_glare'] else 0.0),
        A.Resize(hyperparameter['image_size'], hyperparameter['image_size'], always_apply=True, p=1.0),
        A.CenterCrop(hyperparameter['crop_size'], hyperparameter['crop_size'], always_apply=True, p=1.0),
        A.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)

    hyperparameter_str = str(hyperparameter).replace(', \'', ',\n \'')[1:-1]
    print(f'Hyperparameter info:\n {hyperparameter_str}')
    loaders = prepare_dataset(os.path.join(base_path, ''), hyperparameter, aug_pipeline_train, aug_pipeline_val)

    net = prepare_model(model_path, hyperparameter, device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=hyperparameter['learning_rate'],
                              weight_decay=hyperparameter['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=10, verbose=True)

    desc = f'_paxos_frames_{str("_".join([k[0] + str(hp) for k, hp in hyperparameter.items()]))}'
    writer = SummaryWriter(comment=desc)
    best_model_path = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device, writer, num_epochs=hyperparameter['num_epochs'], description=desc)

    validate(prepare_model(best_model_path, hyperparameter, device), criterion, loaders[1], device, writer, hyperparameter['num_epochs'], calc_roc=True)


def prepare_model(model_path, hp, device):
    # stump = models.alexnet(pretrained=True)
    stump = None
    if hp['multi_channel']:
        stump = my_inceptionv4(pretrained=False)
        hp['pretraining'] = False
    elif hp['narrow_model']:
        stump = NarrowInceptionV1(num_classes=2)
        hp['pretraining'] = False
    else:
        stump = ptm.inceptionv4()
        num_ftrs = stump.last_linear.in_features
        stump.last_linear = nn.Linear(num_ftrs, 2)

    if hp['pretraining']:
        stump.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded stump: ', len(stump.features))
    stump.train()

    for i, child in enumerate(stump.features.children()):
        if i < len(stump.features) * hp['freeze']:
            for param in child.parameters():
                param.requires_grad = False
            dfs_freeze(child)
    stump.to(device)
    return stump


def prepare_dataset(base_name: str, hp, aug_train, aug_val):
    set_names = ('train', 'val') if not hp['preprocessing'] else ('train_pp', 'val_pp')
    if not hp['multi_channel']:
        train_dataset = RetinaDataset(join(base_name, 'labels_train_frames.csv'), join(base_name, set_names[0]), augmentations=aug_train,
                                      balance_ratio=hp['balance'], file_type='', use_prefix=True, boost_frames=hp['boosting'])
        val_dataset = RetinaDataset(join(base_name, 'labels_val_frames.csv'), join(base_name, set_names[1]), augmentations=aug_val, file_type='',
                                    use_prefix=True)
    else:
        train_dataset = MultiChannelRetinaDataset(join(base_name, 'labels_train_frames.csv'), join(base_name, set_names[0]), augmentations=aug_train,
                                                  balance_ratio=hp['balance'], file_type='', use_prefix=True, processed_suffix='_pp')
        val_dataset = MultiChannelRetinaDataset(join(base_name, 'labels_val_frames.csv'), join(base_name, set_names[1]), augmentations=aug_val,
                                                file_type='', use_prefix=True, processed_suffix='_pp')

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=False, sampler=sampler, num_workers=hp['batch_size'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=hp['batch_size'])
    print(f'Dataset info:\n Train size: {len(train_dataset)},\n Validation size: {len(val_dataset)}')
    return train_loader, val_loader


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, num_epochs=50, description='Vanilla'):
    since = time.time()
    best_f1_val = -1
    best_model_path = None

    for epoch in range(num_epochs):
        print(f'{time.strftime("%H:%M:%S")}> Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        cm = torch.zeros(2, 2)

        # Iterate over data.
        for i, batch in tqdm(enumerate(loaders[0]), total=len(loaders[0]), desc=f'Epoch {epoch}'):
            inputs = batch['image'].to(device, dtype=torch.float)
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
        train_scores['loss'] = running_loss / len(loaders[0].dataset)
        write_scores(writer, 'train', train_scores, epoch)
        val_loss, val_f1 = validate(model, criterion, loaders[1], device, writer, epoch)

        if val_f1 > best_f1_val:
            best_f1_val = val_f1
            torch.save(model.state_dict(), f'{time.strftime("%Y%m%d")}_best_paxos_frames_model_{val_f1:0.2}.pth')
            best_model_path = f'{time.strftime("%Y%m%d")}_best_paxos_frames_model_{val_f1:0.2}.pth'

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s with best f1 score of {best_f1_val}')

    return best_model_path


def validate(model, criterion, loader, device, writer, cur_epoch, calc_roc=False) -> Tuple[float, float]:
    model.eval()
    cm = torch.zeros(2, 2)
    running_loss = 0.0
    majority_dict = MajorityDict()

    for i, batch in enumerate(loader):
        inputs = batch['image'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        eye_ids = batch['eye_id']

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)

        for true, pred in zip(labels, preds):
            cm[true, pred] += 1

        majority_dict.add(preds.tolist(), labels, eye_ids)

    scores = calc_scores_from_confusion_matrix(cm)
    scores['loss'] = running_loss / len(loader.dataset)
    if not calc_roc: write_scores(writer, 'val', scores, cur_epoch)

    v = majority_dict.get_predictions_and_labels()
    eye_scores = {'precision': precision_score(v['labels'], v['predictions']),
                  'recall': recall_score(v['labels'], v['predictions']),
                  'f1': f1_score(v['labels'], v['predictions'])}
    if not calc_roc: write_scores(writer, 'eval', eye_scores, cur_epoch)

    if calc_roc:
        write_f1_curve(majority_dict, writer)
        write_pr_curve(majority_dict, writer)

    return running_loss / len(loader.dataset), eye_scores['f1']


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
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
