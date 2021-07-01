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

root_dir= 'data/parent_data/coronal_midline/'
#heatmap_path = f'./{root_dir}/heatmaps/'
mask_path = f'./{root_dir}/cor_mid_mask/'
#coord_path = f'./{root_dir}/coordinates/'
cor_slices_path = f'./{root_dir}/mips/'

#sag_slices_path = './images_sagittal/all_projections/'
#sag_slices_path = './images_sagittal/midline_composite/'

out_root = 'midline_data'
# output paths
train_path = f'./{out_root}/training/'
test_path = f'./{out_root}/testing/'
val_path = f'./{out_root}/validation/'

#Train + (Test/Val)
train_test_split = 0.7

# Test + Val
test_val_split = 0.5

def copy_files(names, out_path):
    print(f'Writing {len(names)} files to {out_path}')
    # Make directories
    os.makedirs(os.path.dirname(
        f'{out_path}slices/coronal/'), exist_ok=True)
    # os.makedirs(os.path.dirname(
    #     f'{out_path}slices/sagittal/'), exist_ok=True)
    # os.makedirs(os.path.dirname(
    #     f'{out_path}targets/heatmaps/'), exist_ok=True)
    os.makedirs(os.path.dirname(
        f'{out_path}targets/masks/'), exist_ok=True)
    # os.makedirs(os.path.dirname(
    #     f'{out_path}targets/coordinates/'), exist_ok=True)

    for name in names:
        filename = name + '.npy'
        csv_filename = name + '.csv'
        # Check if in coronal
        if filename in os.listdir(f'{out_path}slices/coronal/'): 
            pass
        else:
            shutil.copyfile(cor_slices_path + filename, f'{out_path}slices/coronal/{filename}')
        # # Check if sagittal
        # if filename in os.listdir(f'{out_path}slices/sagittal/'):
        #     pass
        # else:
        #     shutil.copyfile(sag_slices_path + filename,
        #                 f'{out_path}slices/sagittal/{filename}')
        # # Check heatmap
        # if filename in os.listdir(f'{out_path}targets/heatmaps/'):
        #     pass
        # else:
        #     shutil.copyfile(heatmap_path + filename,
        #                 f'{out_path}targets/heatmaps/{filename}')
        # Check masks
        if filename in os.listdir(f'{out_path}targets/masks/'):
            pass
        else:
            shutil.copyfile(mask_path + filename,
                            f'{out_path}targets/masks/{filename}')
        # Check coordinates
        # if filename in os.listdir(f'{out_path}targets/coordinates/'):
        #     pass
        # else:
        #     shutil.copyfile(coord_path + csv_filename,
        #                     f'{out_path}targets/coordinates/{csv_filename}')



def main():
    ids = np.array([file.strip('.npy') for file in os.listdir(mask_path)])
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
