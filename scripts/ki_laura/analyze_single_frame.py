import click
import torch
import numpy as np
import cv2
import os
from torchvision import models
from torch import nn
from utils import enhance_contrast_image, get_retina_mask, crop_to_circle
from albumentations.pytorch import ToTensorV2
import albumentations as alb


@click.command()
@click.option('--input_path', '-i', help='Path to the image being analyzed', required=True)
@click.option('--model_path', '-r', help='Path to the trained model', required=True)
@click.option('--processed/--unprocessed', default=False)
def run(input_path, model_path, processed):
    # Loading image
    img = cv2.imread(input_path)
    if (type(img) != np.ndarray and type(img) != np.memmap) or img is None:
        print('Invalid input image: ', input_path)
        return
    if not processed:
        img_enh = enhance_contrast_image(img, clip_limit=3.5, tile_size=12)
        mask, circle = get_retina_mask(img_enh)
        if circle[2] == 0:
            print('Could not detect retinoscope lens.')
            return
        img = cv2.bitwise_and(img, mask)
        img = crop_to_circle(img, circle)
    print('Loaded image successfully...')

    # Necessary image augmentations
    aug_pipeline = alb.Compose([
        alb.Resize(425, 425, always_apply=True, p=1.0),
        alb.CenterCrop(399, 399, always_apply=True, p=1.0),
        alb.Normalize(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0)
    ], p=1.0)
    img_tensor = aug_pipeline(image=img)['image']

    # Loading model
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = models.alexnet()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print('Loaded model successfully...')

    #Prediction
    id_to_class = {0: 'no_retinopathy', 1: 'retinopathy'}
    outputs = model(img_tensor.unsqueeze(0))
    _, prediction = torch.max(outputs, 1)
    prediction = prediction[0].item() 
    print()
    print(f'Prediction for image {os.path.basename(input_path)} with the model {os.path.basename(model_path)}:')
    print(f'------ {id_to_class[prediction]} ------')    


if __name__ == '__main__':
    run()
