"""
Script for splitting data in Train test and validation sets
"""
import numpy as np
import argparse
import sys
import os

# Arguments
shuffle = True
seed = 66

root_dir= 'msk_data'
data_path = f'./{root_dir}/clean_data.npz'
# output paths
train_path = f'./{root_dir}/training_data.npz'
test_path = f'./{root_dir}/test_data.npz'
val_path = f'./{root_dir}/validation_data.npz'

#Train + (Test/Val)
train_test_split = 0.7

# Test + Val
test_val_split = 0.5


def main():
    data = np.load(data_path, allow_pickle=True)
    if shuffle:
        print('Shuffling...., seed:', seed)
    
        np.random.seed(seed=seed)
        indices = np.random.rand(data['slices'].shape[0]).argsort()

        slices = data['slices'][indices]
        masks = data['masks'][indices]
        triplet = data['targets'][indices]
        ids = data['id'][indices]

        # Data sizes
        train_size = int(slices.shape[0]*train_test_split)
        test_size = int((slices.shape[0]-train_size)*test_val_split)
        val_size = int((slices.shape[0]-train_size)*(1-test_val_split))
        print('Train/Test/Val:', train_size, test_size, val_size)
        train_slices, train_masks, train_triplet, train_ids = slices[:
                                                      train_size], masks[:train_size], triplet[:train_size], ids[:train_size]
        test_slices, test_masks, test_triplet, test_ids = slices[train_size:train_size +
                                                   test_size], masks[train_size:train_size+test_size], triplet[train_size:train_size + test_size], ids[train_size:train_size + test_size]
        val_slices, val_masks, val_triplet, val_ids = slices[-val_size:
                                                ], masks[-val_size:], triplet[-val_size:], ids[-val_size:]
        print(train_slices.shape, train_masks.shape)
        np.savez(train_path, slices=train_slices, masks=train_masks,
                 targets=train_triplet, id=train_ids)
        np.savez(test_path, slices=test_slices, masks=test_masks,
                 targets=test_triplet, id=test_ids)
        np.savez(val_path, slices=val_slices, masks=val_masks, 
                 targets=val_triplet, id=val_ids)
    else:
        # Slice arrays
        print(args.size)
        train_slices, train_triplet, train_ids = data['slices'][:
                                                              args.size], data['targets'][:args.size], data['id'][:args.size]
        print(train_slices.shape)
        print(train_ids)
        # Save as npz files
        np.savez(args.output, slices=train_slices,
                 targets=train_triplet, id=train_ids)


if __name__ == '__main__':
    main()
