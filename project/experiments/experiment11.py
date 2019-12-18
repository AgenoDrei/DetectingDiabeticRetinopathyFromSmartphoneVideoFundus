import time

import os
import sys

import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from nn_utils import RandomCrop, RetinaDataset, show_batch, Flip
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

BASE_PATH = '/home/user/mueller9/'
#BASE_PATH = '/data/simon/'


def run():
    writer = SummaryWriter()

    torch.cuda.empty_cache()
    data_transforms = transforms.Compose([
        RandomCrop(299),
        Flip(0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    retina_dataset = RetinaDataset(os.path.join(BASE_PATH, 'combined_retina_dataset.csv'), os.path.join(BASE_PATH, 'combined_retina_dataset'), transform=data_transforms)
    train_size = int(0.9 * len(retina_dataset))
    test_size = len(retina_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(retina_dataset, [train_size, test_size])

    batch_size = 16
    sample_weights = []
    for d in train_dataset:
        if d.label == 1:
            sample_weights.append(1.0)
        sample_weights.append(0.3)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.Tensor(sample_weights).double(), batch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=sampler, num_workers=16)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

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

    model_ft: nn.Module = models.Inception3(num_classes=2)
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, 2)
    #model_ft = model_ft.to(device)

    #weights = np.array([1.0, 2.0])
    #cl_weights = torch.from_numpy(weights).to(device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.9)

    torch.cuda.empty_cache()
    model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loader, device, writer)


def train_model(model, criterion, optimizer, scheduler, loader, device, writer, num_epochs=50):
    since = time.time()
    best_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        model.train()

        running_loss = 0.0
        running_preds = ([], [])

        # Iterate over data.
        for i, batch in enumerate(loader):
            inputs = batch['image'].to(device, dtype=torch.float)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_preds[0].extend(preds.cpu().numpy())
            running_preds[1].extend(labels.cpu().numpy())

        scheduler.step()
            # print('STEP ', i)

        epoch_loss = running_loss / len(loader.dataset)

        writer.add_scalar('Loss/train', epoch_loss)
        writer.add_scalar('Accuracy/train', metrics.accuracy_score(running_preds[1], running_preds[0]))
        writer.add_sclar('ROC/train', metrics.roc_auc_score(running_preds[1], running_preds[0]))
        #print(f'Train Loss: {epoch_loss:.4f} Acc: {metrics.accuracy_score(running_preds[1], running_preds[0]):.4f}, AUC ROC: {metrics.roc_auc_score(running_preds[1], running_preds[0])}')
        #print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #print(f'Best val Acc: {best_acc:4f}')

    torch.save(model.state_dict(), os.path.join(BASE_PATH, f'model{time.time()}.dat'))
    return model


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using opencv version {cv2.__version__}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    run()
    sys.exit(0)
