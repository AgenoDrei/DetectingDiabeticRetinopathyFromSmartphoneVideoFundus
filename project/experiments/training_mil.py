import sys
import cv2
import torch
import argparse
import os
from nn_datasets import get_validation_pipeline, get_training_pipeline, PaxosBags
from nn_models import BagNet
from torch import optim, nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import models


def run(data_path, model_path, gpu_name, batch_size, num_epochs, num_workers):
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')
    hyperparameter = {
        'data': os.path.basename(os.path.normpath(data_path)),
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optim.Adam.__name__,
        'freeze': 0.0,
        'balance': 0.45,
        'image_size': 450,
        'crop_size': 399,
        'pretraining': True,
        'preprocessing': False,
    }
    aug_pipeline_train = get_training_pipeline(hyperparameter['image_size'], hyperparameter['crop_size'])
    aug_pipeline_val = get_validation_pipeline(hyperparameter['image_size'], hyperparameter['crop_size'])

    hyperparameter_str = str(hyperparameter).replace(', \'', ',\n \'')[1:-1]
    print(f'Hyperparameter info:\n {hyperparameter_str}')

    loaders = prepare_dataset(data_path, hyperparameter, aug_pipeline_train, aug_pipeline_val, num_workers)
    net = prepare_network(model_path, hyperparameter, device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=hyperparameter['learning_rate'],
                              weight_decay=hyperparameter['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=10, verbose=True)

    desc = f'_paxos_mil_{str("_".join([k[0] + str(hp) for k, hp in hyperparameter.items()]))}'
    writer = SummaryWriter(comment=desc)

    # train model
    # validate model


def prepare_dataset(data_path, hp, aug_train, aug_val, num_workers):
    train_dataset = PaxosBags('labels_train_frames.csv', os.path.join(data_path, 'train'), augmentations=aug_train,
                              balance_ratio=hp['balance'])
    val_dataset = PaxosBags('labels_val_frames.csv', os.path.join(data_path, 'val'), augmentations=aug_val,
                            balance_ratio=hp['balance'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return [train_loader, test_loader]


def prepare_network(model_path, hp, device):
    stump: models.AlexNet = models.alexnet(pretrained=True)
    if hp['pretraining']:
        stump.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded stump: ', len(stump.features))
    stump.to(device)
    return BagNet(stump)             # Uncool brother of the Nanananannanan Bat-Net


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser(description='Multiple instance learning (sounds like fun)')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--model', help='Path for the base model', type=str)
    parser.add_argument('--gpu', help='GPU name', type=str, default='cuda:0')
    parser.add_argument('--bs', help='Batch size for training', type=int, default=8)
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=50)
    parser.add_argument('--workers', help='Number of workers', type=int, default=16)
    args = parser.parse_args()

    run(args.data, args.model, args.gpu, args.bs, args.epochs, args.workers)
    sys.exit(0)
