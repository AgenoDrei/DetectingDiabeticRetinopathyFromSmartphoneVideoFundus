import os
import sys
import time
from os.path import join
from typing import Tuple

import albumentations as alb
import cv2
import torch
from torch.nn import CrossEntropyLoss, Linear
import torch.optim as optim
from albumentations.augmentations.transforms import RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
import nn_utils
from nn_processing import RandomFiveCrop
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from pretrainedmodels import inceptionv4
import argparse
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm


def run(base_path, gpu_name, batch_size, num_epochs):
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    hyperparameter = {
        'data': os.path.basename(base_path),
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optim.Adam.__name__,
        'image_size': 450,
        'crop_size': 399,
        'freeze': 0.0,
        'balance': 0.25,
        'preprocessing': False
    }
    aug_pipeline_train = alb.Compose([
        alb.Resize(hyperparameter['image_size'], hyperparameter['image_size'], always_apply=True, p=1.0),
        alb.RandomCrop(hyperparameter['crop_size'], hyperparameter['crop_size'], always_apply=True, p=1.0),
        # RandomFiveCrop(hyperparameter['crop_size'], hyperparameter['crop_size'], always_apply=True, p=1.0),
        alb.HorizontalFlip(p=0.5),
        alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        alb.OneOf([alb.GaussNoise(p=0.5), alb.ISONoise(p=0.5), alb.IAAAdditiveGaussianNoise(p=0.25), alb.MultiplicativeNoise(p=0.25)], p=0.3),
        alb.OneOf([alb.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5), alb.GridDistortion(p=0.5)], p=0.25),
        alb.OneOf([alb.HueSaturationValue(p=0.5), alb.ToGray(p=0.5), alb.RGBShift(p=0.5)], p=0.3),
        alb.OneOf([RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2), alb.RandomGamma()], p=0.3),
        alb.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)
    aug_pipeline_val = alb.Compose([
        alb.Resize(hyperparameter['image_size'], hyperparameter['image_size'], always_apply=True, p=1.0),
        alb.CenterCrop(hyperparameter['crop_size'], hyperparameter['crop_size'], always_apply=True, p=1.0),
        alb.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)

    hyperparameter_str = str(hyperparameter).replace(', \'', ',\n \'')[1:-1]
    print(f'Hyperparameter info:\n {hyperparameter_str}')
    loaders = prepare_dataset(os.path.join(base_path, ''), hyperparameter, aug_pipeline_train, aug_pipeline_val)

    net: inceptionv4 = prepare_model(hyperparameter)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=hyperparameter['learning_rate'], weight_decay=hyperparameter['weight_decay'])
    criterion = CrossEntropyLoss()
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.3, patience=7, verbose=True)

    desc = f'_pretraining_{str("_".join([k[0] + str(hp) for k, hp in hyperparameter.items()]))}'
    writer = SummaryWriter(comment=desc)
    train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device, writer, num_epochs=hyperparameter['num_epochs'], description=desc)


def prepare_model(hp):
    net: inceptionv4 = inceptionv4()

    num_ftrs = net.last_linear.in_features
    net.last_linear = Linear(num_ftrs, 2)
    net.train()

    for i, child in enumerate(net.features.children()):
        if i < len(net.features) * hp['freeze']:
            for param in child.parameters():
                param.requires_grad = False
            nn_utils.dfs_freeze(child)
    print(f'Model info: {net.__class__.__name__}, layer: {len(net.features)}, #frozen layer: {len(net.features) * hp["freeze"]}')
    return net


def prepare_dataset(base_name: str, hp, aug_pipeline_train, aug_pipeline_val):
    set_names = ('train2', 'val2') if not hp['preprocessing'] else ('train_pp', 'val_pp')
    train_dataset = nn_utils.RetinaDataset(join(base_name, 'labels_train.csv'), join(base_name, set_names[0]),
                                           augmentations=aug_pipeline_train, balance_ratio=hp['balance'], file_type='.jpg')
    val_dataset = nn_utils.RetinaDataset(join(base_name, 'labels_val.csv'), join(base_name, set_names[1]),
                                                 augmentations=aug_pipeline_val, file_type='.jpg')

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=False, sampler=sampler, num_workers=hp['batch_size'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=hp['batch_size'])
    print(f'Dataset info:\n Train size: {len(train_dataset)},\n Validation size: {len(val_dataset)}')
    return train_loader, val_loader


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, num_epochs=50, description='Vanilla'):
    since = time.time()
    best_f1_val = -1
    model.to(device)

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

            running_loss += loss.item() * inputs.size(0)
            for true, pred in zip(labels, pred):
                cm[int(true), int(pred)] += 1

        train_scores = nn_utils.calc_scores_from_confusion_matrix(cm)
        train_scores['loss'] = running_loss / len(loaders[0].dataset)
        nn_utils.write_scores(writer, 'train', train_scores, epoch)
        val_loss, val_f1 = validate(model, criterion, loaders[1], device, writer, epoch)

        best_f1_val = val_f1 if val_f1 > best_f1_val else best_f1_val

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s with best f1 score of {best_f1_val}')

    validate(model, criterion, loaders[1], device, writer, num_epochs, calc_roc=True)
    torch.save(model.state_dict(), f'model{description}')
    return model


def validate(model, criterion, loader, device, writer, cur_epoch, calc_roc=False) -> Tuple[float, float]:
    model.eval()
    cm = torch.zeros(2, 2)
    running_loss = 0.0
    majority_dict = nn_utils.MajorityDict()
    perf_metrics = nn_utils.Scores()

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Validation'):
        inputs = batch['image'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        # crop_idx = batch['image_idx']
        crop_idx = batch['eye_id']

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)

        majority_dict.add(preds.tolist(), labels, crop_idx)
        perf_metrics.add(preds.tolist(), labels, tags=crop_idx)

    scores = perf_metrics.calc_scores(as_dict=True)
    scores['loss'] = running_loss / len(loader.dataset)
    nn_utils.write_scores(writer, 'val', scores, cur_epoch, full_report=True)

    # print(majority_dict)
    # print(labels, preds)
    #f1_video, recall_video, precision_video = f1_score(labels, preds), recall_score(labels, preds), precision_score(labels, preds)
    #print(f'Validation scores (all 5 crops):\n F1: {f1_video},\n Precision: {precision_video},\n Recall: {recall_video}')
    #writer.add_scalar('val/crof1', f1_video, cur_epoch)

    return running_loss / len(loader.dataset), scores['f1']


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser(description='Pretraing but with crops (glutenfree)')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--gpu', help='GPU name', type=str, default='cuda:0')
    parser.add_argument('--bs', help='Batch size for training', type=int, default=8)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    args = parser.parse_args()

    run(args.data, args.gpu, args.bs, args.epochs)
    sys.exit(0)
