"""
Clean script for training model
"""
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


import albumentations as A
from albumentations.pytorch.transforms import ToTensor




def main():

    pass





if __name__ == '__main__':
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5, limit=5, border_mode=cv2.BORDER_CONSTANT, value=0),
        # ElasticTransform(p=0.5, approximate=True, alpha=75, sigma=8,
        #                  alpha_affine=50, border_mode=cv2.BORDER_CONSTANT, value=0),
        # RandomScale(),
        # Resize(512, 212),
        A.RandomCrop(height=470, width=452, p=0.5),
        A.Resize(626, 452),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(
        #     0.229, 0.224, 0.225), max_pixel_value=1),
        ToTensor()
    ], keypoint_params=A.KeypointParams(format=('yx'), label_fields=['class_labels'], remove_invisible=False), 
    additional_targets={'image1': 'image', 'image2': 'image'})
