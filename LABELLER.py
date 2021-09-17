"""
Script for training vertebrae locator, w/ DSNT & L1 loss
"""

import cv2

import torch
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import segmentation_models_pytorch as smp

from utils.customDataset_v2 import LabelDataset, k_fold_splitter
from argparse import ArgumentParser
import utils.LabellerTL as ltl



torch.autograd.set_detect_anomaly(True)
# *Declare Paths + variables 
parser = ArgumentParser(prog='Run vertebral body segmenter inference')
parser.add_argument(
    '--root_dir', help='Root path containing all folds', type=str)
parser.add_argument(
    '--output_dir', help='Directory to save model + model predictions during inference', type=str)
parser.add_argument(
    '--fold', help='Fold used for !testing! all others used for training', type=int)
parser.add_argument('--mode', help='training/inference',
                    type=str, default='inference')
args = parser.parse_args()

batch_size=4
n_outputs = 13
learning_rate = 3e-3
num_epochs = 500
classifier=True
norm_coords=True
early_stopping=True

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

def main():
    #~Pre-processing + training 
    # ** Create albumentation transforms - train + val + test
    train_transforms = A.Compose([#A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0,
                                 shift_limit=0, p=1, border_mode=0),
            #A.GaussNoise(var_limit=0.025, p=0.5, per_channel=False),
            #A.Perspective(p=0.5),
            A.RandomCrop(height=342, width=512, p=0.5),
            A.Resize(height=512, width=512)
            ], 
        keypoint_params=A.KeypointParams(format=('yx'), label_fields=['labels'], remove_invisible=False))

    valid_transforms = A.Compose([A.Resize(height=512, width=512)],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=True, label_fields=['labels']))

    test_transforms = A.Compose([A.Resize(height=512, width=512)],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=True, label_fields=['labels']), 
        additional_targets={'heatmap': 'mask'})

    #** Pre-processing functions
    pre_processing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    splitter = k_fold_splitter(
        args.root_dir, args.fold, args.mode, num_folds=4)
    
    if args.mode == 'Training':
        #~ Training + val loops
        train, test = splitter.split_data()
        # ** Create Dataset for training
        train_dataset = LabelDataset(
            *train, pre_processing_fn=pre_processing_fn,
            transforms=train_transforms, normalise=True, classifier=classifier, 
            norm_coords=norm_coords)
        valid_dataset = LabelDataset(
            *test, pre_processing_fn=pre_processing_fn,transforms=valid_transforms, 
            normalise=True, classifier=classifier, norm_coords=norm_coords)
        # ** Convert to Dataloaders
        train_generator = DataLoader(train_dataset, batch_size=batch_size)
        valid_generator = DataLoader(valid_dataset, batch_size=batch_size)

        model = ltl.Labeller(training=train_generator, validation=valid_generator, testing=None,
                              dir_name='exp1', n_outputs=14, output_path=args.output_dir, 
                              classifier=classifier, norm_coords=norm_coords, early_stopping=early_stopping)
        model.forward(model_name='labeller.pt',
                      num_epochs=num_epochs)
        #model.train(epoch=0)
        #model.validation(epoch=0)

    elif args.mode == 'Inference':
        #~ Model inference loop
        images, targets = splitter.split_data()
        #* Create dataset for inference
        test_dataset = LabelDataset(
            images, targets, pre_processing_fn=pre_processing_fn,
            transforms=test_transforms, normalise=True, classifier=classifier, norm_coords=norm_coords)
        #** Convert to dataloader
        test_generator = DataLoader(test_dataset, batch_size=1)
        model = ltl.Labeller(training=None, validation=None, testing=test_generator,
                             dir_name='exp1', n_outputs=14, output_path=args.output_dir, 
                             classifier=classifier, norm_coords=norm_coords)
        model.inference(model_name='labeller.pt',
                        plot_output=True, save_preds=True)

    else:
        raise ValueError(
            "Unspecificied mode, should be one of 'Training, Inference'")
    torch.cuda.empty_cache()
         
if __name__ == '__main__':
    main()
