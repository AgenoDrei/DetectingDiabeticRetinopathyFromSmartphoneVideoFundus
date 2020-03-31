import sys
import time
import torch
import argparse
import os
from nn_datasets import get_validation_pipeline, get_training_pipeline, PaxosBags
from nn_models import BagNet
from nn_utils import Scores, write_scores
from torch import optim, nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


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

    best_model_path = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device, writer,
                                  hyperparameter, num_epochs=hyperparameter['num_epochs'], description=desc)
    # validate model


def prepare_dataset(data_path, hp, aug_train, aug_val, num_workers):
    train_dataset = PaxosBags('labels_train_frames.csv', os.path.join(data_path, 'train'), augmentations=aug_train,
                              balance_ratio=hp['balance'])
    val_dataset = PaxosBags('labels_val_frames.csv', os.path.join(data_path, 'val'), augmentations=aug_val,
                            balance_ratio=hp['balance'])

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True,
                              sampler=sampler)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return [train_loader, test_loader]


def prepare_network(model_path, hp, device):
    stump: models.AlexNet = models.alexnet(pretrained=True)
    if hp['pretraining']:
        stump.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded stump: ', len(stump.features))
    stump.to(device)
    return BagNet(stump)  # Uncool brother of the Nanananannanan Bat-Net


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, hp, num_epochs=50,
                description='Vanilla'):
    since = time.time()
    best_f1_val = -1
    best_model_path = None

    for epoch in range(num_epochs):
        print(f'{time.strftime("%H:%M:%S")}> Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        scores = Scores()

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

            scores.add(pred, labels)
            running_loss += loss.item() * inputs.size(0)

        train_scores = scores.calc_scores(as_dict=True)
        train_scores['loss'] = running_loss / len(loaders[0].dataset)
        write_scores(writer, 'train', train_scores, epoch)
        val_loss, val_f1 = validate(model, criterion, loaders[1], device, writer, hp, epoch)

        if val_f1 > best_f1_val:
            best_f1_val = val_f1
            torch.save(model.state_dict(), f'{time.strftime("%Y%m%d")}_best_paxos_frames_model_{val_f1:0.2}.pth')
            best_model_path = f'{time.strftime("%Y%m%d")}_best_paxos_frames_model_{val_f1:0.2}.pth'

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best f1 score: {best_f1_val}, model saved to: {best_model_path}')

    return best_model_path


def validate(model, criterion, loader, device, writer, hp, cur_epoch, calc_roc=False) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    sm = torch.nn.Sigmoid()
    scores = Scores()

    for i, batch in enumerate(loader):
        inputs = batch['image'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        eye_ids = batch['name']

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            probs = sm(outputs)
            running_loss += loss.item() * inputs.size(0)

        scores.add(preds, labels, tags=eye_ids, probs=probs)

    val_scores = scores.calc_scores(as_dict=True)
    val_scores['loss'] = running_loss / len(loader.dataset)
    if not calc_roc: write_scores(writer, 'val', val_scores, cur_epoch)
    return running_loss / len(loader.dataset), val_scores['f1']


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
