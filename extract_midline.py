"""
Use trained midline model to find midline then extract a sagittal midline projection (MIP/AVG)
"""
import os
import torch
from torch.utils.data import DataLoader

import albumentations as A
import segmentation_models_pytorch as smp

import utils.MidlineTL as mtl
from utils.customDataset_v2 import midlineDataset, k_fold_splitter
from utils.midline_utils import *

from argparse import ArgumentParser

torch.autograd.set_detect_anomaly(True)

parser = ArgumentParser(prog='Run midline inference')
parser.add_argument('--root_dir', help='Root path containing all folds', type=str)
parser.add_argument('--volume_dir', help='Parent dir. containing all folds of CT volumes', type=str)
parser.add_argument('--output_dir', help='Directory to save model + model predictions during inference', type=str)
parser.add_argument('--fold', help='Fold used for !testing! all others used for training', type=int)
args = parser.parse_args()

# *Declare Paths + variables
batch_size = 4
n_outputs = 13
learning_rate = 3e-3
num_epochs = 500

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

def main():
    path = os.path.join(args.root_dir, f'q{args.fold}', 'midline_finder_preds.npz')
    data = np.load(path)
    ids, masks = data.values()
    #~ All heavy lifting done in here
    #* Mask to midline, sagittal slice + projection
    extract_sagittal_projections(ids, masks, args.fold, 
                    output_path=args.output_dir, save_slice=True, plot_slice=True)
if __name__ == '__main__':
    main()
