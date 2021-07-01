"""
Script for training Unet to detect midline for coronal projections 
"""
import cv2

import torch
from torch.utils.data import DataLoader

import albumentations as A
import segmentation_models_pytorch as smp

import utils.MidlineTL as Mtl
from utils.customDataset_v2 import midlineDataset

torch.autograd.set_detect_anomaly(True)
# *Declare Paths + variables
train_path = './midline_data/training/'
valid_path = './midline_data/validation/'
test_path = './midline_data/testing/'
batch_size = 4
n_outputs = 13
learning_rate = 3e-3
num_epochs = 200

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

def main():
    #~Pre-processing + training
    # ** Create albumentation transforms - train + val + test
    train_transforms = A.Compose([A.HorizontalFlip(p=0.5),
                                  A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15,
                                  shift_limit=0.1, p=1, border_mode=0),
                                  #A.GaussNoise(var_limit=0.025, p=0.5, per_channel=False),
                                  #A.Perspective(p=0.5),
                                  A.RandomCrop(height=256, width=512, p=0.5),
                                  A.Resize(height=512, width=512)
                                  ],)

    valid_transforms = A.Compose([A.Resize(height=512, width=512)],)

    test_transforms = A.Compose([A.Resize(height=512, width=512)],)

    #** Pre-processing functions
    pre_processing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    # ** Create Dataset for training
    train_dataset = midlineDataset(
        train_path, pre_processing_fn=pre_processing_fn,
        transforms=train_transforms, normalise=True) 
    valid_dataset = midlineDataset(
        valid_path, pre_processing_fn=pre_processing_fn,
        transforms=valid_transforms, normalise=True)
    test_dataset = midlineDataset(
        test_path, pre_processing_fn=pre_processing_fn,
        transforms=test_transforms, normalise=True)

    # ** Convert to Dataloaders
    train_generator = DataLoader(train_dataset, batch_size=batch_size)
    valid_generator = DataLoader(valid_dataset, batch_size=batch_size)
    test_generator = DataLoader(test_dataset, batch_size=1)

    #!! TRAINING + VALIDATION
    model = Mtl.Midline(train_generator, valid_generator, test_generator,
                         dir_name='exp1', detect=True, n_outputs=13)
    model.forward(model_name='midline_finder.pt', num_epochs=300)
    #model.train(epoch=0)
    #model.validation(epoch=0)
    #model.inference(plot_output=True, model_name='midline_finder.pt')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
