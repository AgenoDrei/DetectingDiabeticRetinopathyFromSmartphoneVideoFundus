import os
from os.path import join
import argparse
import pretrainedmodels as ptm
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from nn_datasets import SnippetDataset2, get_training_pipeline, get_validation_pipeline
from nn_models import RetinaNet2
from nn_utils import dfs_freeze, write_scores, \
    Scores
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple
from torchvision import models

def run(base_path, model_path, gpu_name, batch_size, num_epochs):
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    hyperparameter = {
        'data': os.path.basename(os.path.normpath(base_path)),
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optim.Adam.__name__,
        'image_size': 450,
        'crop_size': 399,
        'freeze': 0.0,
        'balance': 0.4,
        'num_frames': 50,
        'pooling': 'avg', # max / avg
        'bag': 'snippet', # snippet / random / snippet sampling
        'pretraining': True,
        'preprocessing': False,
        'stump': 'inception' # alexnet / inception
    }
    aug_pipeline_train = get_training_pipeline(hyperparameter['image_size'], hyperparameter['crop_size'])
    aug_pipeline_val = get_validation_pipeline(hyperparameter['image_size'], hyperparameter['crop_size'])

    hyperparameter_str = str(hyperparameter).replace(', \'', ',\n \'')[1:-1]
    print(f'Hyperparameter info:\n {hyperparameter_str}')
    loaders = prepare_dataset(os.path.join(base_path, ''), hyperparameter, aug_pipeline_train, aug_pipeline_val)

    net: RetinaNet2 = prepare_model(model_path, hyperparameter, device)

    optimizer_ft = optim.Adam(net.parameters(), lr=hyperparameter['learning_rate'], weight_decay=hyperparameter['weight_decay'])
    optimizer_ft = optim.Adam([{'params': net.features.parameters(), 'lr': 1e-5},
                               {'params': net.pooling.parameters()},
                               {'params': net.features2.parameters()},
                               {'params': net.after_pooling.parameters()}], lr=hyperparameter['learning_rate'],
                              weight_decay=hyperparameter['weight_decay'])

    criterion = nn.CrossEntropyLoss()
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=15, verbose=True)

    desc = f'_video_{str("_".join([k[0] + str(hp) for k, hp in hyperparameter.items()]))}'
    writer = SummaryWriter(comment=desc)
    model = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device, writer,
                        num_epochs=hyperparameter['num_epochs'], description=desc)


def prepare_model(model_path, hp, device):
    stump = None
    
    if hp['stump'] == 'inception':
        stump = ptm.inceptionv4()
        num_ftrs = stump.last_linear.in_features
        stump.last_linear = nn.Linear(num_ftrs, 2)
        for i, child in enumerate(stump.features.children()):
            if i < len(stump.features) * hp['freeze']:
                for param in child.parameters():
                    param.requires_grad = False
                dfs_freeze(child)
    elif hp['stump'] == 'alexnet':
        stump = models.alexnet(pretrained=True)
        num_ftrs = stump.classifier[-1].in_features
        stump.classifier[-1] = nn.Linear(num_ftrs, 2)
    
    if hp['pretraining']:
        stump.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded stump: ', len(stump.features))
    stump.train()
    net = RetinaNet2(frame_stump=stump, pooling_strategy=hp['pooling'], stump_type=hp['stump'])
    return net


def prepare_dataset(base_name: str, hp, aug_pipeline_train, aug_pipeline_val):
    set_names = ('train', 'val') if not hp['preprocessing'] else ('train_pp', 'val_pp')
    # Get labels file regardless for pipeline version
    csv_files = os.listdir(base_name)
    csv_files = [c for c in csv_files if c.endswith('.csv') and c.startswith('labels')]
    labels_train = sorted([c for c in csv_files if 'train' in c], key=lambda s: len(s), reverse=True)[0]
    labels_val = sorted([c for c in csv_files if 'val' in c], key=lambda s: len(s), reverse=True)[0]
    print('Found label files: ', labels_train, labels_val)

    train_dataset = SnippetDataset2(join(base_name, labels_train), join(base_name, set_names[0]),
                                    augmentations=aug_pipeline_train, balance_ratio=hp['balance'],
                                    num_frames=hp['num_frames'], bagging_strategy=hp['bag'])
    val_dataset = SnippetDataset2(join(base_name, labels_val), join(base_name, set_names[1]),
                                  augmentations=aug_pipeline_val, num_frames=hp['num_frames'], bagging_strategy=hp['bag'])

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                               sampler=sampler, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=16)
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
        scores = Scores()

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
            scores.add(pred, labels)

        train_scores = scores.calc_scores(as_dict=True)
        train_scores['loss'] = running_loss / len(loaders[0].dataset)
        write_scores(writer, 'train', train_scores, epoch)
        val_loss, val_f1 = validate(model, criterion, loaders[1], device, writer, epoch)

        best_f1_val = val_f1 if val_f1 > best_f1_val else best_f1_val

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(
        f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s with best f1 score of {best_f1_val}')

    validate(model, criterion, loaders[1], device, writer, num_epochs, calc_roc=True)
    torch.save(model.state_dict(), f'model{description}')
    return model


def validate(model, criterion, loader, device, writer, cur_epoch, calc_roc=False) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    scores = Scores()

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Validation'):
        inputs = batch['frames'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        video_name = batch['name']

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)

        scores.add(preds, labels, tags=video_name)

    val_scores = scores.calc_scores(as_dict=True)
    val_scores['loss'] = running_loss / len(loader.dataset)
    write_scores(writer, 'val', val_scores, cur_epoch)

    video_scores = scores.calc_scores_eye(as_dict=True)
    write_scores(writer, 'eval', video_scores, cur_epoch)

    return running_loss / len(loader.dataset), val_scores['f1']


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser(description='Classify a paxos dataset through snippet learning')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--model', help='Path for the base model', type=str)
    parser.add_argument('--gpu', help='GPU name', type=str, default='cuda:0')
    parser.add_argument('--bs', help='Batch size for training', type=int, default=8)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    args = parser.parse_args()
    print('INFO> ', args)

    run(args.data, args.model, args.gpu, args.bs, args.epochs)
    sys.exit(0)
