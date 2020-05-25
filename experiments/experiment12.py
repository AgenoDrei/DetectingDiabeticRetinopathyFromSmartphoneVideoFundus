from pytorch_lightning import Trainer
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils import data
from torch.optim import lr_scheduler
from torchvision import models
import pretrainedmodels as ptm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from nn_utils import RetinaDataset
from collections import OrderedDict

class RetinaSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(RetinaSystem, self).__init__()
        self.hparams = hparams

        self.net = models.resnet50(pretrained=True)
        # freeze net
        children = self.net.children()
        for i, child in enumerate(children):
            if i < hparams.freeze_factor * len(list(children)):
                for param in child.parameters():
                    param.require_grad = False
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 2)

    def training_step(self, batch, batch_idx):
        inputs = batch['image']
        labels = batch['label']

        out = self.net.forward(inputs)
        _, preds = torch.max(out, 1)

        loss = nn.functional.cross_entropy(out, labels)
        logger_logs = {'training/loss': loss}

        return {'loss': loss, 'progress_bar': {'training/loss': loss}, 'log': logger_logs}

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        labels = batch['label']

        out = self.net.forward(inputs)
        _, preds = torch.max(out, 1)

        loss = nn.functional.cross_entropy(out, labels)
        cm = torch.zeros(2, 2)
        for true, pred in zip(labels, preds):
            cm[true, pred] += 1

        output = OrderedDict({
            'val_loss': loss.item(),
            'cm': cm
        })
        return output

    def validation_end(self, outputs):
        val_loss_mean = 0
        cm = torch.zeros(2, 2)

        for output in outputs:
            val_loss_mean += output['val_loss']
            cm += output['cm']
        val_loss_mean /= len(outputs)

        tp = cm[1, 1].item()
        fp = cm[0, 1].item()
        fn = cm[1, 0].item()
        precision = tp / (tp + fp + 0.1)
        recall = tp / (tp + fn + 0.1)
        f1 = (2 * precision * recall) / (precision + recall + 0.1)

        tqdm_dict = {'val/loss': val_loss_mean, 'val/f1': f1, 'val/precision': precision, 'val/recall': recall}
        logger_logs = {'val/loss': val_loss_mean, 'val/f1': f1, 'val/precision': precision, 'val/recall': recall}
        results = {
            'progress_bar': tqdm_dict,
            'log': logger_logs
        }
        return results


    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
        return [opt], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        aug_pipeline_train = A.Compose([
            A.Resize(self.hparams.image_size, self.hparams.image_size, always_apply=True, p=1.0),
            A.RandomCrop(self.hparams.crop_size, self.hparams.crop_size, always_apply=True, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(min_holes=1, max_holes=4, max_width=100, max_height=100, min_width=25, min_height=25, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.OneOf([A.HueSaturationValue(p=0.5), A.ToGray(p=0.5), A.RGBShift(p=0.5), A.RandomGamma(p=0.5)], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(always_apply=True, p=1.0)
        ], p=1.0)

        train_dataset = RetinaDataset(os.path.join(self.hparams.path, f'retina_labels_train.csv'), os.path.join(self.hparams.path, f'retina_data_train'),
                                      augmentations=aug_pipeline_train, file_type='.jpg', balance_ratio=0.25)
        sample_weights = [train_dataset.get_weight(i) for i in range(len(train_dataset))]
        sampler = data.sampler.WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
        train_loader = data.DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=False, sampler=sampler, num_workers=16)
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        aug_pipeline_val = A.Compose([
            A.Resize(self.hparams.image_size, self.hparams.image_size, always_apply=True, p=1.0),
            A.CenterCrop(self.hparams.crop_size, self.hparams.crop_size, always_apply=True, p=1.0),
            A.Normalize(always_apply=True, p=1.0),
            ToTensorV2(always_apply=True, p=1.0)
        ], p=1.0)
        val_dataset = RetinaDataset(os.path.join(self.hparams.path, f'retina_labels_val.csv'), os.path.join(self.hparams.path, f'retina_data_val'),
                                    augmentations=aug_pipeline_val, file_type='.jpg')
        val_loader = data.DataLoader(val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=16)
        return val_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--image_size', default=300, type=int)
        parser.add_argument('--crop_size', default=299, type=int)
        parser.add_argument('--freeze_factor', default=0, type=float)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser


def main(hparams):
    # init module
    model = RetinaSystem(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        distributed_backend='ddp',
        nb_gpu_nodes=hparams.nodes,
        show_progress_bar=True,
        default_save_path=f'{os.getcwd()}/runs_lightning'

    )
    trainer.fit(model)
    print(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
    print('and going to http://localhost:6006 on your browser')


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--path', type=str, default=None)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = RetinaSystem.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)