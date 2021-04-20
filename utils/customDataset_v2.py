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
import albumentations as A


class spineDataset(Dataset):
    def __init__(self, dir_path, pre_processing_fn=None, transforms=None, normalise=True):
        #~Custom dataset for spine models with coronal and sagittal inputs 
        super(spineDataset, self).__init__()
        
        self.coronal_inputs = self.load_data(dir_path + 'slices/coronal/')
        self.sagittal_inputs = self.load_data(dir_path + 'slices/sagittal/')
        self.heatmaps = self.load_data(dir_path + 'targets/heatmaps/')
        #//self.masks = self.load_data(dir_path + 'targets/masks/')
        self.coordinates =  self.load_coordinates(dir_path + 'targets/coordinates/')
        self.norm_coronal_inputs = self.normalise_inputs(self.coronal_inputs)
        self.norm_sagittal_inputs = self.normalise_inputs(self.sagittal_inputs)#
        self.pre_processing_fn = self.get_preprocessing(pre_processing_fn)
        self.ids = self.get_ids()
        self.transforms = transforms
        self.normalise = normalise
        self.img_size = self.coronal_inputs['slices'].shape[1:3]
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']

    def get_ids(self):
        #* Get patient names
        assert self.coronal_inputs['id'] == self.sagittal_inputs['id'] == self.heatmaps['id']
        ids = self.coronal_inputs['id']
        return ids

    @staticmethod
    def load_data(path):
        #*Load directory of npy slices.
        data_dict = {'slices': [], 'id': []}
        for file in os.listdir(path):
            if file.endswith('.npy'):
                name = file.split('.')[0]
                data_dict['id'].append(name)
                slice_ = np.load(path + file)
                data_dict['slices'].append(slice_)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict
    
    @staticmethod
    def load_coordinates(path):
        #*Load directory of csv files with coordinate info
        data_dict = {}
        for file in os.listdir(path):
            if file.endswith('.csv'):
                name = file.split('.')[0]
                df = pd.read_csv(path + file)
                data_dict[name] = {}
                for i, row in df.iterrows():
                    data_dict[name][row[0]] = row[1]
        return data_dict

    @staticmethod
    def normalise_inputs(data):
        #*Normalise inputs between [0, 1]
        return (data['slices'] - data['slices'].min())/(data['slices'].max()-data['slices'].min())

    def convert2keypoints(self, pid):
        #*Convert 1D coordinate into 2D keypoints + labels (one-hot)
        coord_dict = self.coordinates[pid]
        keypoints = []
        for vert in self.ordered_verts:
            if vert in coord_dict.keys():
                keypoints.append([coord_dict[vert], 226])
            else:
                keypoints.append([0, 0])
        
        keypoints = np.array(keypoints)
        labels = [vert in coord_dict.keys() for vert in self.ordered_verts]
        labels = np.array(labels).astype(int)
        return keypoints, labels

    @staticmethod
    def keypoints2tensor(keypoints):
        #*Convert into tensor (BxCx1)
        tgt_list = []
        for elem in keypoints:
            y,x = elem
            coord = torch.tensor(y)
            tgt_list.append(coord)
        return torch.stack(tgt_list, dim=0)

    @staticmethod
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self, pre_processing_fn):
        """
        Construct preprocessing transform
        from https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb 
        [Cell 11]
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose
        """
        _transform = [
            A.Lambda(name='pre_process', image=pre_processing_fn),
            A.Lambda(name='to_tensor', image=self.to_tensor, mask=self.to_tensor),
        ]
        return A.Compose(_transform)

    def __len__(self):
        return len(self.coronal_inputs['id'])

    def __getitem__(self, index):
        #~ Main Function
        pid = self.ids[index]
        
        if self.normalise:
            #* Norm inputs -> [0, 1]
            sag_img = self.norm_sagittal_inputs[index]
            cor_img = self.norm_coronal_inputs[index]
        else:
            sag_img = self.sagittal_inputs['slices'][index]
            cor_img = self.coronal_inputs['slices'][index]
        heatmap = self.heatmaps['slices'][index]

        #// mask = np.argmax(self.masks['slices'][index], axis=-1)
        keypoints, labels = self.convert2keypoints(pid)
        if self.transforms:
            #* Apply transforms
            augmented = self.transforms(
                image=sag_img, image1=cor_img, mask=heatmap, keypoints=keypoints, class_labels=labels)
            sag_img, cor_img, heatmap, keypoints, labels = augmented.values()
            #* Pre-processing for using segmentation-models library (w/ pre-trained encoders)
            if self.pre_processing_fn is not None:
                sag_prep = self.pre_processing_fn(image=sag_img, mask=heatmap)
                cor_prep = self.pre_processing_fn(image=cor_img)
                sag_img, heatmap, cor_img = sag_prep['image'], sag_prep['mask'], cor_prep['image']
            #* Convert keypoints to tensor
            out_keypoints = self.keypoints2tensor(keypoints)
            #* Convert list of labels to tensor
            out_labels = torch.stack([torch.tensor(elem) for elem in labels], dim=0)

            #~Prepare sample
            sample = {'sag_image': sag_img, 
            'cor_image': cor_img,
            'heatmap': np.mean(heatmap, axis=2),
            'keypoints': out_keypoints,
            'class_labels': out_labels,
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
