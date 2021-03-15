"""
Script containing custom dataset for training triplet classifier 
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import SimpleITK as sitk

class threeChannelDataset(Dataset):
    def __init__(self, fname, transforms=None, normalise=True, singleLevel=False):
        """
        Custom dataset with Window Level settings instead of dataset-wide rescaling.
        """
        super(threeChannelDataset, self).__init__()
        self.data = np.load(fname, allow_pickle=True)
        self.transforms = transforms
        self.inputs = self.data['slices'][..., np.newaxis]
        self.in_tgts = self.data['targets']
        self.patient_ids = self.data['id']
        self.transforms = transforms
        self.normalise = normalise
        self.img_size = self.inputs.shape[1:3]
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']
        self.norm_inputs = (self.inputs-self.inputs.min()) / \
            (self.inputs.max()-self.inputs.min())
        self.filter_mask = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if singleLevel:
            self.masks = self.single_level_prep(level='L3')
        else:
            self.masks = np.argmax(np.squeeze(self.data['masks'], axis=-1))# (252, ~600, ~400, 14)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.normalise:
            img = self.norm_inputs[index]
        else:
            img = self.inputs[index]
        targets = self.in_tgts[index]
        labels = [key for key in targets.keys()]
        # Order targets + convert to tensors
        targets = [tuple(val) for val in targets.values()]
        img = self.convert_threeChannel(img)
        if self.transforms:
            
            augmented = self.transforms(image=img, keypoints=targets, mask=self.masks[index], class_labels=labels)
            aug_img, aug_tgt, aug_mask, aug_labels = augmented[
                'image'], augmented['keypoints'], augmented['mask'], augmented['class_labels']
            # Normalise between [-1, 1]
            # Present_tnsr = binary vector showing presence of verts
            #aug_tgt, present_tnsr= self.norm_tgt(aug_tgt, aug_labels)
            aug_tgt = self.norm_tgt_singleLevel(aug_tgt, aug_labels, level='L3')
            aug_tgt = aug_tgt.unsqueeze(0)
            sample = {'inputs': aug_img, 'targets': aug_tgt,
                    'masks': aug_mask,
                    'class_labels': self.ordered_verts,
                    'present': self.filter_mask,
                    'id': self.patient_ids[index]}
        else:
            print('Need some transforms - minimum convert to Tensor')
            return
        return sample

    def single_level_prep(self, level='T12'):
        """
        Radius in mm 
        Assumes inputs have been resampled to 0.3125 and downsampled by 1/4 i.e. pixel size = 1.25mm
        """
        channel = self.ordered_verts.index(level)
        masks = np.squeeze(self.data['masks'])[..., channel+1]
        return np.where(masks != 0, 1, 0)  # Out shape (batch, H, W, 1)
       

    def write_patient_info(self, filename):
        """
        Write patient ids to text file
        """
        with open(filename, 'w') as f:
            f.write(f'Training Set Size: {len(self.patient_ids)}\n')
            f.write('Patient IDs: \n')
            for i, id in enumerate(self.patient_ids):
                f.write(f'{i}: {id}\n')

    def convert_threeChannel(self, img):
        # Convert to list to enable index assignment
        img_shape_list = list(img.shape)
        img_shape_list[-1] = 3
        threeChannelImg = np.zeros((tuple(img_shape_list)))
        threeChannelImg[..., 0] = img[..., 0]
        threeChannelImg[..., 1] = img[..., 0]
        threeChannelImg[..., 2] = img[..., 0]
        return threeChannelImg

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

    def norm_tgt(self, targets, labels):
        out_list = []
        presence_list = []
        for vert in self.ordered_verts:
            if vert in labels:
                idx = labels.index(vert)
                elem = targets[idx]
                coords = torch.tensor(elem)
                norm_coords = (coords * 2 + 1) / \
                    torch.Tensor(self.img_size) - 1
                out_list.append(norm_coords)
                presence_list.append(1)
            else:
                coords = torch.Tensor((0.0, 0.0))
                out_list.append(coords)
                presence_list.append(0)       
        out_tensor = torch.stack(out_list, dim=0)
        out_present = torch.tensor(presence_list)
        return out_tensor, out_present
        
    def norm_tgt_singleLevel(self, targets, labels, level='T12'):

        if level in labels:
            idx = labels.index(level)
            elem = targets[idx]
            coords = torch.tensor(elem)
            norm_coords = (coords * 2 + 1) / \
                torch.Tensor(self.img_size) - 1
            return norm_coords
        else:
            return torch.Tensor((0.0, 0.0))
