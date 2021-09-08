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
from scipy.special import softmax
import dsntnn
import scipy.stats as ss

from einops import rearrange

class spineDataset(Dataset):
    def __init__(self, dir_path, pre_processing_fn=None, transforms=None, normalise=True, sample_points=False, detect=False):
        #~Custom dataset for spine models with coronal and sagittal inputs 
        super(spineDataset, self).__init__()
        
        self.coronal_inputs = self.load_data(dir_path + 'slices/coronal/')
        self.sagittal_inputs = self.load_data(dir_path + 'slices/sagittal/')
        self.heatmaps = self.load_data(dir_path + 'targets/heatmaps/')
        self.masks = self.load_data(dir_path + 'targets/masks/')
        self.coordinates =  self.load_coordinates(dir_path + 'targets/coordinates/')
        self.norm_coronal_inputs = self.normalise_inputs(self.coronal_inputs)
        self.norm_sagittal_inputs = self.normalise_inputs(self.sagittal_inputs)
        self.norm_heatmaps = self.normalise_inputs(self.heatmaps)
        self.pre_processing_fn = self.get_preprocessing(pre_processing_fn)
        self.ids = self.get_ids()
        self.transforms = transforms
        self.normalise = normalise
        self.sample_points = sample_points
        self.detect = detect
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

    def keypoints2tensor(self, keypoints, size):
        #*Convert into tensor (BxCx1)
        tgt_list = []
        for elem in keypoints:
            y,x = elem
            coord = torch.tensor(y)
            tgt_list.append(coord)
        tensor = torch.stack(tgt_list, dim=0)
        #* Normalise between [-1, 1]
        return dsntnn.pixel_to_normalized_coordinates(tensor, size=size)

    @staticmethod
    def to_tensor(x, **kwargs):
        return rearrange('h w c -> c h w', x).astype(np.float32)
        #return x.transpose(2, 0, 1).astype('float32')

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

    @staticmethod
    def sample_keypoints(keypoints, sigma=25):
        sampled_points = []
        for point in keypoints:
            norm = ss.norm(loc=point[0], scale=sigma)
            x = norm.rvs()
            #print('Point/RVS', point[0], x)
            coord = [x, point[1]]
            sampled_points.append(coord)
        out_points = np.array(sampled_points)
        return out_points


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
        #* Apply softmax across each channel of heatmap
       #// heatmap = self.norm_heatmaps[index]
        heatmap = self.heatmaps['slices'][index]
        if self.detect:
            mask = self.masks['slices'][index]
        keypoints, labels = self.convert2keypoints(pid)
        #!! Don't randomly sample keypoints when doing validation
        if self.sample_points:
            keypoints = self.sample_keypoints(keypoints, sigma=3)
        if self.transforms:
            #* Apply transforms
            augmented = self.transforms(
                image=sag_img, image1=cor_img, mask=heatmap, keypoints=keypoints, class_labels=labels, mask1=mask)
            sag_img, cor_img, heatmap, keypoints, labels, mask = augmented.values()
            #* Pre-processing for using segmentation-models library (w/ pre-trained encoders)
            if self.pre_processing_fn is not None:
                sag_prep = self.pre_processing_fn(image=sag_img, mask=heatmap)
                cor_prep = self.pre_processing_fn(image=cor_img, mask=mask)
                sag_img, heatmap, cor_img, mask = sag_prep['image'], sag_prep['mask'], cor_prep['image'], cor_prep['mask']
            #* Convert keypoints to tensor
            out_keypoints = self.keypoints2tensor(keypoints, size=sag_img.shape[1])
            #* Convert list of labels to tensor
            out_labels = torch.stack([torch.tensor(elem) for elem in labels], dim=0)
            out_heatmap = np.mean(heatmap, axis=2)
            out_mask = torch.max(torch.tensor(mask), dim=0, keepdim=True).values
            #~Prepare sample
            sample = {'sag_image': sag_img, 
            'cor_image': cor_img,
            'heatmap': out_heatmap,
            'keypoints': out_keypoints,
            'class_labels': out_labels,
            'mask': out_mask,
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

##@ ------------------------------------------------------------------------- MIDLINE DATASET ---------------------------------------------

class midlineDataset(Dataset):
    def __init__(self, img_paths, tgt_paths, pre_processing_fn=None, transforms=None, normalise=True):
        #~Custom dataset for spine models with coronal and sagittal inputs
        super().__init__()

        self.coronal_inputs = self.load_data(img_paths)
        self.masks = self.load_data(tgt_paths)
        self.norm_coronal_inputs = self.normalise_inputs(self.coronal_inputs)
        self.pre_processing_fn = self.get_preprocessing(pre_processing_fn)
        self.ids = self.get_ids()
        self.transforms = transforms
        self.normalise = normalise
        self.img_size = self.coronal_inputs['slices'].shape[1:3]
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']

    def get_ids(self):
        #* Get patient names
        return self.coronal_inputs['id']

    @staticmethod
    def load_data(path_dict):
        #*Load directory of npy slices.
        data_dict = {'slices': [], 'id': []}
        for pid, path in path_dict.items():
            data_dict['id'].append(pid)
            slice_ = np.load(path)
            data_dict['slices'].append(slice_)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict

    @staticmethod
    def normalise_inputs(data):
        #*Normalise inputs between [0, 1]
        return (data['slices'] - data['slices'].min())/(data['slices'].max()-data['slices'].min())

    @staticmethod
    def to_tensor(x, **kwargs):
        return rearrange('h w c -> c h w', x).astype(np.float32)

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
            A.Lambda(name='to_tensor', image=self.to_tensor,
                     mask=self.to_tensor),
        ]
        return A.Compose(_transform)

    def __len__(self):
        return len(self.coronal_inputs['id'])

    def __getitem__(self, index):
        #~ Main Function
        pid = self.ids[index]
        if self.normalise:
            #* Norm inputs -> [0, 1]
            cor_img = self.norm_coronal_inputs[index][..., np.newaxis]
        else:
            cor_img = self.coronal_inputs['slices'][index][..., np.newaxis]
        #* Apply softmax across each channel of heatmap
        
        mask = self.masks['slices'][index]

        #!! Don't randomly sample keypoints when doing validation
        if self.transforms:
            #* Apply transforms
            augmented = self.transforms(image=cor_img, mask=mask)
            cor_img, mask = augmented.values()
            #* Pre-processing for using segmentation-models library (w/ pre-trained encoders)
            if self.pre_processing_fn is not None:
                cor_prep = self.pre_processing_fn(image=cor_img, mask=mask)
                cor_img, mask = cor_prep['image'], cor_prep['mask']
            #* Convert list of labels to tensor
            out_mask = torch.max(torch.tensor(
                mask), dim=0, keepdim=True).values
            #~Prepare sample
            sample = {'cor_image': cor_img,
                      'mask': out_mask,
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


#@ ------------------------------------------------------SEGMENTER DATASET --------------------------------------------------------

class SegmenterDataset(Dataset):
    def __init__(self, img_paths, tgt_paths, pre_processing_fn=None, transforms=None, normalise=True):
        #~ Custom Dataset for sagittal vert. body segmenter
        super().__init__()
        self.sagittal_inputs = self.load_data(img_paths)
        self.masks = self.load_data(tgt_paths)
        self.norm_sagittal_inputs = self.normalise_inputs(self.sagittal_inputs)
        self.pre_processing_fn = self.get_preprocessing(pre_processing_fn)
        self.ids = self.get_ids()
        self.transforms = transforms
        self.normalise = normalise
        self.img_size = self.sagittal_inputs['slices'].shape[1:3]
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']

    @staticmethod
    def load_data(path_dict):
        #*Load directory of npy slices.
        data_dict = {'slices': [], 'id': []}
        for pid, path in path_dict.items():
            data_dict['id'].append(pid)
            slice_ = np.load(path)
            data_dict['slices'].append(slice_)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict

    @staticmethod
    def normalise_inputs(data):
        #*Normalise inputs between [0, 1]
        return (data['slices'] - data['slices'].min())/(data['slices'].max()-data['slices'].min())

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
            A.Lambda(name='to_tensor', image=self.to_tensor,
                     mask=self.to_tensor),
        ]
        return A.Compose(_transform)

    @staticmethod
    def to_tensor(x, **kwargs):
        return rearrange(torch.tensor(x), 'h w c -> c h w').to(dtype=torch.float32)
    
    def get_ids(self):
        #* Get patient names
        return self.sagittal_inputs['id']

    def __len__(self):
        return len(self.sagittal_inputs['id'])
    
    def __getitem__(self, index):
        #~ Main Function
        pid = self.ids[index]
        if self.normalise:
            #* Norm inputs -> [0, 1]
            sag_img = self.norm_sagittal_inputs[index][..., np.newaxis]
        else:
            sag_img = self.sagittal_inputs['slices'][index][..., np.newaxis]
        #* Apply softmax across each channel of heatmap

        mask = self.masks['slices'][index][..., np.newaxis]
        if self.transforms:
            #* Apply transforms
            augmented = self.transforms(image=sag_img, mask=mask)
            sag_img, mask = augmented.values()
            #* Pre-processing for using segmentation-models library (w/ pre-trained encoders)
            if self.pre_processing_fn is not None:
                sag_prep = self.pre_processing_fn(image=sag_img, mask=mask)
                sag_img, mask = sag_prep['image'], sag_prep['mask']
            #~Prepare sample
            sample = {'sag_image': sag_img,
                      'mask': torch.squeeze(mask),
                      'id': pid}
        else:
            print('Need some transforms - minimum convert to Tensor')
            return
        return sample

###@ ---------------------------------------------------------- K FOLD SPLITTER ------------------------------------------------------

class k_fold_splitter():
    def __init__(self, root_dir, testing_fold, mode, num_folds=4):
        self.root = root_dir
        self.test_fold = testing_fold
        self.mode = mode
        self.seed = 12345
        self.train_test_ratio=0.8
    
    def split_data(self):
        if self.mode == 'Inference':
            return self.inference_split()
        if self.mode == 'Training':
            return self.training_split()
        else:
            raise ValueError(
                "Unspecificied mode, should be one of 'Training, Inference'")


    def inference_split(self):
        #~ go through testing fold and return relevant filepaths
        tgt_dict = {}
        img_dict = {}
        for root, dirs, files in os.walk(self.root):
            if files and f'q{self.test_fold}' in root:
                if 'slices' in root:
                    for file in files:
                        name = file.split('.')[0]
                        img_dict[name] = os.path.join(root, file)
                elif 'targets' in root:
                    for file in files:
                        name = file.split('.')[0]
                        tgt_dict[name] = os.path.join(root, file)
                else:
                    continue
        assert len(img_dict.keys()) != 0 and len(tgt_dict.keys()) !=0, 'Issue with directory structure'
        return img_dict, tgt_dict

    def training_split(self):
        #~ Collect all training files, then train/val split and return both lists
        img_dict = {}
        tgt_dict = {}
        for root, dirs, files in os.walk(self.root):
            if files and f'q{self.test_fold}' not in root:
                if 'slices' in root:
                    for file in files:
                        name = file.split('.')[0]
                        img_dict[name] = os.path.join(root, file)
                elif 'targets' in root:
                    for file in files:
                        name = file.split('.')[0]
                        tgt_dict[name] = os.path.join(root, file)
                else:
                    continue
        assert len(img_dict.keys()) != 0 and len(tgt_dict.keys()) !=0, 'Issue with directory structure'
        assert img_dict.keys() == tgt_dict.keys(), "Keys don't match for masks and images"
        train_ids, test_ids = self.train_test_ids(img_dict)
        train_img, train_tgt = self.split_dicts(train_ids, img_dict, tgt_dict)
        test_img, test_tgt = self.split_dicts(test_ids, img_dict, tgt_dict)
        return (train_img, train_tgt), (test_img, test_tgt)

    @staticmethod
    def split_dicts(ids, img_dict, tgt_dict):
        return {pid: img_dict[pid] for pid in ids}, {pid: tgt_dict[pid] for pid in ids}

    def train_test_ids(self, img_dict):
        #~ Shuffle and pick ids for training/testing
        ids = list(img_dict.keys())
        np.random.seed(seed=self.seed)
        indices = np.random.rand(len(ids)).argsort()
        train_size = int(len(ids)*self.train_test_ratio)
        print(train_size, len(ids), len(indices))
        train_ids = np.array(ids)[indices[:train_size]]
        test_ids = np.array(ids)[indices[train_size:]]
        print(f'Training: {len(train_ids)}; Testing: {len(test_ids)}')
        return train_ids, test_ids
