import time

import os
import sys

import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from nn_utils import RandomNormalCrop, EnhanceContrast, RetinaDataset, show_batch, Flip, Blur
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import joblib as job

BASE_PATH = '/home/user/mueller9/'
#BASE_PATH = '/data/simon/'
GPU_ID = 'cuda:2'
BATCH_SIZE = 32

def run():
    writer = SummaryWriter()

    torch.cuda.empty_cache()
    data_transforms = transforms.Compose([
        RandomNormalCrop(448),
        EnhanceContrast(0.75),
        Flip(0.5),
        Blur(0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    retina_dataset = RetinaDataset(os.path.join(BASE_PATH, 'combined_retina_dataset.csv'), os.path.join(BASE_PATH, 'combined_retina_dataset'), transform=data_transforms)
    train_size = int(0.95 * len(retina_dataset))
    test_size = len(retina_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(retina_dataset, [train_size, test_size])

    sample_weights = [retina_dataset.get_weight(i) for i in train_dataset.indices]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=16)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
    device = torch.device(GPU_ID if torch.cuda.is_available() else "cpu")

    print(f'Dataset info:\n Train size: {train_size},\n Test size: {test_size},\n Device: {device}')

    model_ft: nn.Module = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    #model_ft = model_ft.to(device)

    weights = np.array([1.0, 1.0])
    cl_weights = torch.from_numpy(weights).to(device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=cl_weights)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay=0.0001)
    
    #cyclic_scheduler = lr_scheduler.CyclicLR(optimizer_ft, 0.000001, 0.0001, step_size_up=1000, gamma=0.9, mode='exp_range', cycle_momentum=False)
    step_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.5)

    torch.cuda.empty_cache()
    model = train_model(model_ft, criterion, optimizer_ft, step_scheduler, [train_loader, val_loader], device, writer)


def train_model(model, criterion, optimizer, scheduler, loaders, device, writer, num_epochs=100):
    since = time.time()
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
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

        scheduler.step() # Change depending on scheduling!
        # print('STEP ', i)
        print(running_loss / len(loaders[0].dataset))
        print(cm)
        writer.add_scalar('train/loss', running_loss / len(loaders[0].dataset), epoch)
        if epoch % 10 == 9:
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
    for i, batch in enumerate(loader):
        inputs = batch['image'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        for true, pred in zip(labels, preds):
            cm[true, pred] += 1

    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()

    precision = tp / (tp + fp + 0.1)
    recall = tp / (tp + fn + 0.1)
    f1 = 2 * (precision * recall) / (precision + recall + 0.1)

    writer.add_scalar('train/f1', f1, cur_epoch)
    writer.add_scalar('train/precision', precision, cur_epoch)
    writer.add_scalar('train/recall', recall, cur_epoch)
    print(cm)
    print(f'Scores:\n F1: {f1},\n Precision: {precision},\n Recall: {recall}')

if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    run()
    sys.exit(0)
