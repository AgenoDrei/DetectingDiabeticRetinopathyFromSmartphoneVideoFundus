import sys
import nn_utils
import torch
import argparse
import albumentations as alb
import pretrainedmodels as ptm
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils import data
from os.path import join
from tqdm import tqdm


def run(data_path, model_path, gpu_name, bs):
    image_size = 450
    crop_size = 399
    aug_pipeline = alb.Compose([
        alb.Resize(image_size, image_size, always_apply=True, p=1.0),
        alb.CenterCrop(crop_size, crop_size, always_apply=True, p=1.0),
        alb.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
    net = prepare_model(model_path, device)
    data_loader = prepare_dataset(data_path, aug_pipeline, bs)

    make_predictions(net, data_loader, device)


def prepare_model(model_path, device):
    stump = ptm.inceptionv4()   # if not hp['multi_channel'] else my_inceptionv4(pretrained=False)

    num_ftrs = stump.last_linear.in_features
    stump.last_linear = nn.Linear(num_ftrs, 2)
    stump.load_state_dict(torch.load(model_path, map_location=device))
    return stump


def prepare_dataset(data_path, aug_pipeline, bs):
    dataset = nn_utils.RetinaDataset(join(data_path, 'labels_train_frames.csv'), data_path, augmentations=aug_pipeline, file_type='.png')
    loader = data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=16)
    print(f'Dataset size: {len(dataset)}')
    return loader


def make_predictions(model, loader, device):
    model.eval()
    cm = torch.zeros(2, 2)

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Validation'):
        inputs = batch['image'].to(device, dtype=torch.float)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        for true, pred in zip(labels, preds):
            cm[true, pred] += 1

    scores = nn_utils.calc_scores_from_confusion_matrix(cm)
    print(f'Scores:\n F1: {scores["f1"]},\n Precision: {scores["precision"]},\n Recall: {scores["recall"]}')


if __name__ == '__main__':
    print(f'INFO> Using python version {sys.version_info}')
    print(f'INFO> Using torch with GPU {torch.cuda.is_available()}')

    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--data', help='Path for training data', type=str)
    parser.add_argument('--model', help='Path for the base model', type=str)
    parser.add_argument('--labels', help='Output file for modified labels', type=str)
    parser.add_argument('--gpu', help='GPU name', type=str, default='cuda:0')
    parser.add_argument('--bs', help='Batch size for training', type=int, default=8)
    args = parser.parse_args()

    run(args.data, args.model, args.gpu, args.bs)
    sys.exit(0)
