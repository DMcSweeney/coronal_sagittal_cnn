"""
Script for training a CNN for producing heatmaps of vertebra centre points.
"""
import os
from tqdm import tqdm
import numpy as np

import dsntnn

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchsummary import summary
from torch.optim import Adam
import torch.nn as nn

import albumentations as A
from albumentations.pytorch.transforms import ToTensor

from utils.customDataset_v2 import spineDataset
from utils.customWriter_v2 import customWriter
from utils.customModel import customResnet
import cv2

torch.autograd.set_detect_anomaly(True)
train_path = './data/training/'
valid_path = './data/validation/'
output_path = './outputs/'

batch_size = 4
input_channels = 1
num_outputs = 13
max_epochs = 200
learning_rate = 3e-3
# Loss weights
alpha=10
beta = 0.00001

device = torch.device("cuda:0")
torch.cuda.set_device(device)

def train(train_generator, valid_generator):
    model = customResnet(n_outputs=num_outputs, input_size=(626, 452))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', verbose=True)
    
    # Applies LogSoftmax internally
    criterion = nn.MSELoss().cuda()
    #ce_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 57.9, 55.3, 55.4, 56.1, 48.8, 38.8,
    #                                                     29.4, 24.2, 24.9, 27.4, 28.9, 29.4, 30.3])).cuda()


    writer = customWriter(log_dir=output_path,
                          batch_size=batch_size, epoch=0, num_classes=num_outputs)

    best_loss = 100000
    improved_epochs = []
    # Use same training data for x repetitions
    for epoch in range(max_epochs+1):
        writer.epoch = epoch
        print('Epoch: {}/{}'.format(epoch, max_epochs))
        # Slowly switch from one loss to another over course of training
        model.train()
        writer.reset_losses()
        for idx, data in enumerate(tqdm(train_generator)):
            sag_img, cor_img = data['sag_image'].to(
                device, dtype=torch.float32), data['cor_image'].to(device, dtype=torch.float32)
            heatmaps = data['heatmap'].to(device, dtype=torch.float32)
            heatmaps = torch.squeeze(heatmaps.permute(0, 1, 3, 2))
            keypoints, labels = data['keypoints'], data['class_labels']           
            one_hot = data['one-hot'] # Vector encoding presence of a vert in the image

            print(keypoints, labels)
            # Remove single channel 
            if epoch % 25==0 and idx==0:
                writer.plot_inputs(f'Sagittal Inputs at epoch {epoch}', sag_img, targets=[keypoints, labels])
                writer.plot_inputs(
                    f'Coronal Inputs at epoch {epoch}', cor_img)
                writer.plot_histogram(f'Target heatmap at epoch {epoch}', heatmaps, labels)

            optimizer.zero_grad()
           
            pred_heatmaps, seg_pred = model(sag_img, cor_img)
            pred= torch.squeeze(pred_heatmaps)
            mse_loss = criterion(pred, heatmaps)
            loss = mse_loss
            
            writer.train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            break
        break
                 
        print('Train Loss:', np.mean(writer.train_loss))
        writer.add_scalar('training_loss', np.mean(writer.train_loss), epoch)
        loss_list = []
        with torch.set_grad_enabled(False):
            print('VALIDATION')
            for batch_idx, data in enumerate(valid_generator):
                sag_img, cor_img = data['sag_image'].to(
                    device, dtype=torch.float32), data['cor_image'].to(device, dtype=torch.float32)
                heatmaps = data['heatmap'].to(device, dtype=torch.float32)
                heatmaps = torch.squeeze(heatmaps.permute(0, 1, 3, 2))
                labels = data['labels']
                # Remove single channel
                optimizer.zero_grad()
                
                pred_heatmaps, seg_pred = model(sag_img, cor_img)
                pred = torch.squeeze(pred_heatmaps)
                mse_loss = criterion(pred, heatmaps)
                if epoch % 25 == 0 and batch_idx == 0:
                    writer.plot_histogram(f'Prediction at epoch {epoch}', heatmaps, labels=labels, prediction=pred_heatmaps)
                
                valid_loss = mse_loss

                loss_list.append(valid_loss.item())
                writer.val_loss.append(valid_loss.item())
                

            loss1 = np.mean(writer.val_loss)
            is_best = loss1 < best_loss
            best_loss = min(loss1, best_loss)
            if is_best:
                print('Saving best model')
                torch.save(model.state_dict(),
                           output_path + 'best_model.pt')
                improved_epochs.append(epoch)
            print('Val Loss:', np.mean(writer.val_loss))
            scheduler.step(np.mean(writer.val_loss))
            
            
            writer.add_scalar('validation loss',
                              np.mean(writer.val_loss), epoch)

    # Write improved epochs to txt file so I can find when the last best model was saved
    with open('Improvement_epoch.txt', 'w') as f:
        f.write('Epochs where val loss was at a minimum:\n')
        for item in improved_epochs:
            f.write(f'{item}\n')
    writer.close()
    return model

if __name__ =='__main__':
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5, limit=5, border_mode=cv2.BORDER_CONSTANT, value=0),
        # ElasticTransform(p=0.5, approximate=True, alpha=75, sigma=8,
        #                  alpha_affine=50, border_mode=cv2.BORDER_CONSTANT, value=0),
        # RandomScale(),
        # Resize(512, 212),
        A.RandomCrop(height=470, width=452, p=0.5),
        A.Resize(626, 452),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(
        #     0.229, 0.224, 0.225), max_pixel_value=1),
        ToTensor()
    ], keypoint_params=A.KeypointParams(format=('yx'), label_fields=['class_labels']), additional_targets={'image1': 'image', 'image2': 'image'})

    valid_transforms = A.Compose([
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1), 
        ToTensor()], keypoint_params=A.KeypointParams(format=('yx'), label_fields=['class_labels']), additional_targets={'image1': 'image', 'image2': 'image'})

    train_dataset = spineDataset(train_path, transforms=train_transforms, normalise=True)
    valid_dataset = spineDataset(valid_path, transforms=valid_transforms, normalise=True)
   
    # train_dataset.write_categories('overview.txt')
    # valid_dataset.write_categories('overview_valid.txt')

    train_generator = DataLoader(train_dataset, batch_size=batch_size)
    valid_generator = DataLoader(valid_dataset, batch_size=batch_size)

    model = train(train_generator, valid_generator)
    # Save model
    torch.save(model.state_dict(), output_path + 'model.pt')
    torch.cuda.empty_cache()
