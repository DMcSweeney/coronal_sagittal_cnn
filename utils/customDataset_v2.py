"""
Script containing custom dataset for training vertebral labelling model
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import SimpleITK as sitk


class spineDataset(Dataset):
    def __init__(self, dir_path, transforms=None, normalise=True):
        """
        Custom dataset for spine models with coronal and sagittal inputs 
        """
        super(spineDataset, self).__init__()
        
        self.coronal_inputs = self.load_data(dir_path + 'slices/coronal/')
        self.sagittal_inputs = self.load_data(dir_path + 'slices/sagittal/')
        self.heatmaps = self.load_data(dir_path + 'targets/heatmaps/')
        #self.masks = self.load_data(dir_path + 'targets/masks/')
        self.coordinates =  self.load_coordinates(dir_path + 'targets/coordinates/')
        self.norm_coronal_inputs = self.normalise_inputs(self.coronal_inputs)
        self.norm_sagittal_inputs = self.normalise_inputs(self.sagittal_inputs)
        self.ids = self.get_ids()
        self.transforms = transforms
        self.normalise = normalise
        self.img_size = self.coronal_inputs['slices'].shape[1:3]
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']
        # self.filter_mask = torch.tensor(
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    def get_ids(self):
        assert self.coronal_inputs['id'] == self.sagittal_inputs['id'] == self.heatmaps['id']
        ids = self.coronal_inputs['id']
        return ids


    def load_data(self, path):
        """
        Load directory of npy slices.
        """
        data_dict = {'slices': [], 'id': []}
        for file in os.listdir(path):
            if file.endswith('.npy'):
                name = file.split('.')[0]
                data_dict['id'].append(name)
                slice_ = np.load(path + file)
                data_dict['slices'].append(slice_)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict
    
    def load_coordinates(self, path):
        """
        Load directory of csv files with coordinate info
        """
        data_dict = {}
        for file in os.listdir(path):
            if file.endswith('.csv'):
                name = file.split('.')[0]
                df = pd.read_csv(path + file)
                data_dict[name] = {}
                for i, row in df.iterrows():
                    data_dict[name][row[0]] = row[1]
        return data_dict

    def normalise_inputs(self, data):
        """
        Normalise inputs between [0, 1]
        """
        return (data['slices'] - data['slices'].min())/(data['slices'].max()-data['slices'].min())

    def convert2keypoints(self, pid):
        coord_dict = self.coordinates[pid]
        keypoints = [(y,226)  for y in coord_dict.values()]
        labels = [str(vert) for vert in coord_dict.keys()]
        return keypoints, labels


    def __len__(self):
        return len(self.coronal_inputs['id'])

    def __getitem__(self, index):
        pid = self.ids[index]
        
        if self.normalise:
            # Norm inputs -> [0, 1]
            sag_img = self.norm_sagittal_inputs[index]
            cor_img = self.norm_coronal_inputs[index]
        else:
            sag_img = self.sagittal_inputs['slices'][index]
            cor_img = self.coronal_inputs['slices'][index]
        # Convert heatmap to 1
        heatmap = self.heatmaps['slices'][index]
        #mask = np.argmax(self.masks['slices'][index], axis=-1)
        keypoints, labels = self.convert2keypoints(pid)

        # Order targets + convert to tensors
        if self.transforms:
            augmented = self.transforms(
                image=sag_img, image1=cor_img, mask=heatmap, keypoints=keypoints, class_labels=labels)
            aug_sag_img, aug_cor_img, aug_heatmap, aug_keypoints, aug_labels = augmented.values()
            # One-hot vector indicating presence of vert
            one_hot_vert = [vert in aug_labels for vert in self.ordered_verts]
            one_hot_vert = np.array(one_hot_vert).astype(int)
            # Convert heatmap to one dim
            sample = {'sag_image': aug_sag_img, 
            'cor_image': aug_cor_img,
            'heatmap': torch.mean(aug_heatmap, dim=2),
            'keypoints': aug_keypoints,
            'class_labels': aug_labels,
            'one-hot': one_hot_vert,
            'id': pid}
        else:
            print('Need some transforms - minimum convert to Tensor')
            return
        return sample

    def write_patient_info(self, filename):
        """
        Write patient ids to text file
        """
        with open(filename, 'w') as f:
            f.write(f'Training Set Size: {len(self.patient_ids)}\n')
            f.write('Patient IDs: \n')
            for i, id in enumerate(self.patient_ids):
                f.write(f'{i}: {id}\n')

    def write_categories(self, filename):
        """
        Write mapping from triplet names to one hot encoding to a text file
        """
        arg_labels = [int(x) for x in np.asarray(self.targets)]
        counts = Counter(arg_labels)
        with open(filename, 'w') as f:
            f.write(f'Vert encoding: {self.categories}\n')
            for i, cat in enumerate(self.categories):
                f.write(f'{cat}: {counts[i]}\n')
