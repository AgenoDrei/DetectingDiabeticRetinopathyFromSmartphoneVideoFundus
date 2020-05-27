import argparse
import os
import sys
import time
from os.path import join
from typing import Tuple
import albumentations as A
import cv2
import pretrainedmodels as ptm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.augmentations.transforms import RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from multichannel_inceptionv4 import my_inceptionv4
from narrow_inceptionv import NarrowInceptionV1
from nn_datasets import RetinaDataset, MultiChannelRetinaDataset
from nn_processing import ThresholdGlare
from nn_utils import dfs_freeze, write_scores, Scores
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import models


def run(base_path, model_path, gpu_name, batch_size, num_epochs, num_workers):
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    hyperparameter = {
        'data': os.path.basename(os.path.normpath(base_path)),
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optim.Adam.__name__,
        'freeze': 0.0,
        'balance': 0.3,
        'image_size': 450,
        'crop_size': 399,
        'pretraining': True,
        'preprocessing': False,
        'multi_channel': False,
        'boosting': 1.00,
        'use_clahe': False,
        'narrow_model': False,
        'remove_glare': False,
        'voting_percentage': 1.0,
        'validation': 'tv', # tvt = train / val / test, tt = train(train + val) / test, tv = train / val
        'network': 'alexnet' # alexnet / inception
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
    loaders = prepare_dataset(os.path.join(base_path, ''), hyperparameter, aug_pipeline_train, aug_pipeline_val,
                              num_workers)
    net = prepare_model(model_path, hyperparameter, device)

    optimizer_ft = optim.Adam([{'params': net.features.parameters(), 'lr': 1e-5},
                               {'params': net.classifier.parameters()}],
            lr=hyperparameter['learning_rate'], weight_decay=hyperparameter['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=10, verbose=True)

    desc = f'_paxos_frames_{str("_".join([k[0] + str(hp) for k, hp in hyperparameter.items()]))}'
    writer = SummaryWriter(comment=desc)
    model_path, f1 = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device, writer, hyperparameter, num_epochs=hyperparameter['num_epochs'], description=desc)
    print('Best identified model: ', model_path)
    print('Performance F1: ', f1)
    return f1
    # validate(prepare_model(best_model_path, hyperparameter, device), criterion, loaders[2], device, writer, hyperparameter, hyperparameter['num_epochs'], is_test=True)


def prepare_model(model_path, hp, device):
    # stump = models.alexnet(pretrained=True)
    stump = None
    if hp['multi_channel']:
        stump = my_inceptionv4(pretrained=False)
        hp['pretraining'] = False
    elif hp['narrow_model']:
        stump = NarrowInceptionV1(num_classes=2)
        hp['pretraining'] = False
    elif hp['network'] == 'inception':
        stump = ptm.inceptionv4()
        num_ftrs = stump.last_linear.in_features
        stump.last_linear = nn.Linear(num_ftrs, 2)
    elif hp['network'] == 'alexnet':
        stump = models.alexnet(pretrained=True)
        stump.classifier[-1] = nn.Linear(stump.classifier[-1].in_features, 2)

    if hp['pretraining']:
        stump.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded stump: ', len(stump.features))
    stump.train()

    #for i, child in enumerate(stump.features.children()):
    #    if i < len(stump.features) * hp['freeze']:
    #        for param in child.parameters():
    #            param.requires_grad = False
    #        dfs_freeze(child)
    stump.to(device)
    return stump


def prepare_dataset(base_name: str, hp, aug_train, aug_val, num_workers):
    set_names = {'train': 'train', 'val': 'val', 'test': 'test'}
    if hp['preprocessing']:
        set_names = {'train': 'train_pp', 'val': 'val_pp'}
    elif hp['validation'] == 'tt':
        set_names = {'train': 'train+val', 'val': 'test', 'test': 'test'}
    elif hp['validation'] == 'tv':
        set_names = {'train': 'train', 'val': 'val', 'test': 'val'}

    if not hp['multi_channel']:
        train_dataset = RetinaDataset(join(base_name, f'labels_{set_names["train"]}_frames.csv'), join(base_name, set_names['train']),
                                      augmentations=aug_train,
                                      balance_ratio=hp['balance'], file_type='', use_prefix=True,
                                      boost_frames=hp['boosting'], occur_balance=False)
        val_dataset = RetinaDataset(join(base_name, f'labels_{set_names["val"]}_frames.csv'), join(base_name, set_names['val']),
                                    augmentations=aug_val, file_type='',
                                    use_prefix=True)
        test_dataset = RetinaDataset(join(base_name, f'labels_{set_names["test"]}_frames.csv'), join(base_name, set_names['test']),
                                    augmentations=aug_val, file_type='',
                                    use_prefix=True)
    else:
        train_dataset = MultiChannelRetinaDataset(join(base_name, 'labels_train_frames.csv'),
                                                  join(base_name, set_names.train), augmentations=aug_train,
                                                  balance_ratio=hp['balance'], file_type='', use_prefix=True,
                                                  processed_suffix='_pp')
        val_dataset = MultiChannelRetinaDataset(join(base_name, 'labels_val_frames.csv'), join(base_name, set_names.val),
                                                augmentations=aug_val,
                                                file_type='', use_prefix=True, processed_suffix='_pp')

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=False,
                                               sampler=sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False,
                                             num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False,
                                            num_workers=num_workers)
    print(f'Dataset info:\n Train size: {len(train_dataset)},\n Validation size: {len(val_dataset)}')
    return train_loader, val_loader, test_loader


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, hp, num_epochs=50, description='Vanilla'):
    since = time.time()
    best_f1_val = -1
    best_model_path = None

    for epoch in range(num_epochs):
        print(f'{time.strftime("%H:%M:%S")}> Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        scores = Scores()

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

            scores.add(pred, labels)
            running_loss += loss.item() * inputs.size(0)

        train_scores = scores.calc_scores(as_dict=True)
        train_scores['loss'] = running_loss / len(loaders[0].dataset)
        write_scores(writer, 'train', train_scores, epoch)
        val_loss, val_f1 = validate(model, criterion, loaders[1], device, writer, hp, epoch)
        if hp['validation'] == 'tvt': validate(model, criterion, loaders[2], device, writer, hp, epoch, is_test=True)

        if val_f1 > best_f1_val:
            best_f1_val = val_f1
            best_model_path = f'best_frames_model_f1_{val_f1:0.2}_epoch_{epoch}.pth'
            torch.save(model.state_dict(), best_model_path)

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s with best f1 score of {best_f1_val}')

    if hp['validation'] == 'tt' or hp['validation'] == 'tv': # For test set evaluation, the last model should be used (specified by number of epochs) therefore overwrite the best_model_path
        best_model_path = f'frames_model_evalf1_{val_f1:0.2}_epoch_{epoch}.pth'
        torch.save(model.state_dict(), best_model_path)
    return best_model_path, val_f1


def validate(model, criterion, loader, device, writer, hp, cur_epoch, is_test=False) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    sm = torch.nn.Softmax(dim=1)
    scores = Scores()

    for i, batch in enumerate(loader):
        inputs = batch['image'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        eye_ids = batch['eye_id']

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = sm(outputs)
            running_loss += loss.item() * inputs.size(0)

        scores.add(preds, labels, tags=eye_ids, probs=probs)

    val_scores = scores.calc_scores(as_dict=True)
    val_scores['loss'] = running_loss / len(loader.dataset)

    eye_scores = scores.calc_scores_eye(as_dict=True, top_percent=hp['voting_percentage'])
    if not is_test:
        write_scores(writer, 'val', val_scores, cur_epoch)
        write_scores(writer, 'eval', eye_scores, cur_epoch)
    else:
        write_scores(writer, 'test', val_scores, cur_epoch)
        write_scores(writer, 'etest', eye_scores, cur_epoch)

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
    parser.add_argument('--workers', help='Number of workers', type=int, default=16)
    args = parser.parse_args()

    run(args.data, args.model, args.gpu, args.bs, args.epochs, args.workers)
    sys.exit(0)
