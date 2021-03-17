"""
Script for splitting data in Train test and validation sets
adapted from TrainTestValSplit.py to use folders of .npy files instead of .npz files

Output structure: data -> training/validation/testing -> slices -> coronal/sagittal
                                                        -> targets
"""
import numpy as np
import argparse
import sys
import os
import shutil

# Arguments
shuffle = True
seed = 66

root_dir= 'data'
data_path = f'./{root_dir}/heatmaps/'
cor_slices_path = './images_coronal/all_projections/'
sag_slices_path = './images_sagittal/all_projections/'


# output paths
train_path = f'./{root_dir}/training/'
test_path = f'./{root_dir}/testing/'
val_path = f'./{root_dir}/validation/'

#Train + (Test/Val)
train_test_split = 0.7

# Test + Val
test_val_split = 0.5

def copy_files(names, out_path):
    print(f'Writing {len(names)} files to {out_path}')
    for name in names:
        filename = name + '.npy'
        if filename in os.listdir(f'{out_path}slices/coronal/'): continue
        shutil.copyfile(cor_slices_path + filename, f'{out_path}slices/coronal/{filename}')
        shutil.copyfile(sag_slices_path + filename,
                        f'{out_path}slices/sagittal/{filename}')
        shutil.copyfile(data_path + filename,
                        f'{out_path}targets/{filename}')


def main():
    ids = np.array([file.strip('.npy') for file in os.listdir(data_path)])
    print('Shuffling...., seed:', seed)

    np.random.seed(seed=seed)
    indices = np.random.rand(len(ids)).argsort()
    # Data sizes
    train_size = int(len(ids)*train_test_split)
    test_size = int((len(ids)-train_size)*test_val_split)
    val_size = int((len(ids)-train_size)*(1-test_val_split))
    print('TRAIN/TEST/VAL', train_size, test_size, val_size)
    train_ids = ids[indices[:train_size]]
    val_ids = ids[indices[-val_size:]]
    test_ids = ids[indices[train_size:train_size+test_size]]
    copy_files(train_ids, train_path)
    copy_files(val_ids, val_path)
    copy_files(test_ids, test_path)


if __name__ == '__main__':
    main()
