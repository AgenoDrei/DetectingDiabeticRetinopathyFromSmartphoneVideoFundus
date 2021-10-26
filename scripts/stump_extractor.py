import argparse
import os
import time
import torch
from torch import nn
from torchvision import models
from nn_models import BagNet


def run(model_path):
    stump = models.alexnet(pretrained=True)
    num_features = stump.classifier[-1].in_features
    stump.classifier[-1] = nn.Linear(num_features, 2) 

    model =  net = BagNet(stump, num_attention_neurons=738, attention_strategy='normal', pooling_strategy='max', stump_type='alexnet') 
    model.load_state_dict(torch.load(model_path))
    torch.save(model.stump.state_dict(), f'{time.strftime("%Y%m%d")}_stump_extracted.pth')


if __name__ == '__main__': 
    a = argparse.ArgumentParser(description='Extract stump from full model export')
    a.add_argument("--model_path", help="absolute path to input model")
    args = a.parse_args()

    run(args.model_path)
