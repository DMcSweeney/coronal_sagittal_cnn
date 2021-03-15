"""
Script for testing trained models
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models as models
from customDataset import threeChannelDataset
from customModel import customResnet

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensor

training_path = './outputs/'
model_name = 'best_model.pt'
test_data = './msk_data/test_data.npz'
batch_size = 1
num_outputs = 13

device = torch.device("cuda:0")
torch.cuda.set_device(device)

def inference(test_generator):
    #model = models.segmentation.fcn_resnet101(pretrained=False, num_classes=num_outputs).to(device)

    model = customResnet(n_outputs=num_outputs)
    # Change to single channel inputs
    # model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    model.load_state_dict(torch.load(training_path + model_name))
    model.to(device)
    model.eval()
    id_list = []
    with torch.set_grad_enabled(False):
        for idx, data in enumerate(test_generator):
            inputs, id_ = data['inputs'].to(
                device, dtype=torch.float32), data['id']
            targets = data['targets']
            #targets = targets[0].permute(0, 3, 1, 2)

            #targets = torch.argmax(targets, axis=1)
            id_list.append(id_)
            coords, heatmaps, unnorm_heatmaps, seg_pred = model(inputs)
            if idx == 0:
                input_slices = inputs
                predictions = seg_pred
                out_coords = coords
                out_targets = targets

            else:
                input_slices = torch.cat((input_slices, inputs), dim=0)
                predictions = torch.cat((predictions, seg_pred), dim=0)
                out_targets = torch.cat((out_targets, targets), dim=0)
                out_coords = torch.cat((out_coords, coords), dim=0)
    return input_slices, predictions, id_list, out_targets, out_coords


if __name__ == '__main__':
    test_transforms = Compose([Normalize(mean=(0.485, 0.456, 0.406), std=(
        0.229, 0.224, 0.225), max_pixel_value=1),
                               ToTensor()])
    test_dataset = threeChannelDataset(
        test_data, transforms=test_transforms, normalise=False)
    print('Test Size', test_dataset.__len__())
    test_generator = DataLoader(test_dataset, batch_size)
    input_slices, predictions, id_list, targets, coords = inference(test_generator)
    np.savez(training_path + 'predictions.npz', slices=input_slices.cpu(),
             masks=predictions.cpu(), id=id_list, targets=targets, coords=coords.cpu())


