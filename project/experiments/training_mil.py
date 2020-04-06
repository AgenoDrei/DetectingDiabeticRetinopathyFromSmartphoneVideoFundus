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
from typing import Tuple


def run(data_path, model_path, gpu_name, batch_size, num_epochs, num_workers):
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')
    hyperparameter = {
        'data': os.path.basename(os.path.normpath(data_path)),
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optim.Adam.__name__,
        'freeze': 0.0,
        'balance': 0.45,
        'image_size': 450,
        'crop_size': 399,
        'pretraining': True,
        'preprocessing': False,
        'attention_neurons': 1024,
        'bag_size': 50,
        'attention': 'normal',          # normal / gated
        'pooling': 'max'                # avg / max / none
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
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=15, verbose=True)

    desc = f'_paxos_mil_{str("_".join([k[0] + str(hp) for k, hp in hyperparameter.items()]))}'
    writer = SummaryWriter(comment=desc)

    best_model_path = train_model(net, criterion, optimizer_ft, plateau_scheduler, loaders, device, writer,
                                  hyperparameter, num_epochs=hyperparameter['num_epochs'], description=desc)
    validate(prepare_network(best_model_path, hyperparameter, device, whole_net=True), criterion, loaders[1], device, writer, hyperparameter, hyperparameter['num_epochs'], calc_roc=True)


def prepare_dataset(data_path, hp, aug_train, aug_val, num_workers):
    print('Preparing dataset...')
    train_dataset = PaxosBags(os.path.join(data_path, 'labels_train_frames.csv'), os.path.join(data_path, 'train'), augmentations=aug_train,
                              balance_ratio=hp['balance'], max_bag_size=hp['bag_size'])
    val_dataset = PaxosBags(os.path.join(data_path, 'labels_val_frames.csv'), os.path.join(data_path, 'val'), augmentations=aug_val, max_bag_size=hp['bag_size'])

    sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                              sampler=sampler)
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return [train_loader, test_loader]


def prepare_network(model_path, hp, device, whole_net=False):
    print('Preparing network...')
    stump: models.AlexNet = models.alexnet(pretrained=True)
    num_features = stump.classifier[-1].in_features
    stump.classifier[-1] = nn.Linear(num_features, 2)
    if hp['pretraining'] and not whole_net:
        stump.load_state_dict(torch.load(model_path, map_location=device))
        print('Loaded stump: ', len(stump.features))
    stump.to(device)
    net = BagNet(stump, num_attention_neurons=hp['attention_neurons'], attention_strategy=hp['attention'],
                 pooling_strategy=hp['pooling'])  # Uncool brother of the Nanananannanan Bat-Net
    if whole_net:
        net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    return net


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, hp, num_epochs=50,
                description='Vanilla'):
    print('Training model...')
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
            label = batch['label'].to(device)

            model.train()
            optimizer.zero_grad()

            loss, _ = model.calculate_objective(inputs, label)
            error, pred = model.calculate_classification_error(inputs, label)
            
            loss.backward()
            optimizer.step()

            scores.add(pred, label)
            running_loss += loss.item()

        train_scores = scores.calc_scores(as_dict=True)
        train_scores['loss'] = running_loss / len(loaders[0].dataset)
        write_scores(writer, 'train', train_scores, epoch)
        val_loss, val_f1 = validate(model, criterion, loaders[1], device, writer, hp, epoch)

        if val_f1 > best_f1_val:
            best_f1_val = val_f1
            torch.save(model.stump.state_dict(), f'{time.strftime("%Y%m%d")}_best_paxos_frames_model_{val_f1:0.2}.pth')
            best_model_path = f'{time.strftime("%Y%m%d")}_best_paxos_frames_model_{val_f1:0.2}.pth'

        scheduler.step(val_loss)

    time_elapsed = time.time() - since
    print(f'{time.strftime("%H:%M:%S")}> Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best f1 score: {best_f1_val}, model saved to: {best_model_path}')

    return best_model_path


def validate(model, criterion, loader, device, writer, hp, cur_epoch, calc_roc=False) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    scores = Scores()

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Validation'):
        inputs = batch['frames'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        eye_ids = batch['name']

        with torch.no_grad():
            loss, attention_weights = model.calculate_objective(inputs, labels)
            error, preds = model.calculate_classification_error(inputs, labels) 
            running_loss += loss.item()

        scores.add(preds, labels, tags=eye_ids)

    val_scores = scores.calc_scores(as_dict=True)
    val_scores['loss'] = running_loss / len(loader.dataset)
    if not calc_roc:
        write_scores(writer, 'val', val_scores, cur_epoch)
        eye_scores = scores.calc_scores_eye(as_dict=True)
        write_scores(writer, 'eval', eye_scores, cur_epoch)
    else:
        writer.add_hparams(hparam_dict=hp, metric_dict=val_scores)
    
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
