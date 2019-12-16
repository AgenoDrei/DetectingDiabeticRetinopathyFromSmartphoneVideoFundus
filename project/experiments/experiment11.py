import time

import os
import sys

import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from nn_utils import RandomCrop, RetinaDataset, show_batch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = '/data/simon/'


def run():
    torch.cuda.empty_cache()
    data_transforms = transforms.Compose([
        RandomCrop(600),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    retina_dataset = RetinaDataset('/data/simon/processed_trainLabels.csv', '/data/simon/processed_retina_data', transform=data_transforms)
    train_size = int(0.9 * len(retina_dataset))
    test_size = len(retina_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(retina_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Dataset info:\n Train size: {train_size},\n Test size: {test_size},\n Device: {device}')

    # for i_batch, sample_batched in enumerate(train_loader):
    #     print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
    #     if i_batch == 3:
    #         plt.figure()
    #         show_batch(sample_batched)
    #         plt.axis('off')
    #         plt.ioff()
    #         plt.show()
    #         break

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)

    weights = np.array([0.25, 1.0])
    cl_weights = torch.from_numpy(weights).to(device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=cl_weights)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    torch.cuda.empty_cache()
    model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loader, device)


def train_model(model, criterion, optimizer, scheduler, loader, device, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, batch in enumerate(loader):
            inputs = batch['image'].to(device, dtype=torch.float)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

            scheduler.step()
            # print('STEP ', i)

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects / len(loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    torch.save(model.state_dict(), os.path.join(BASE_PATH, f'model{time.time()}.dat'))
    return model


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    run()
    sys.exit(0)