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
from albumentations import Compose, HorizontalFlip, Rotate, ElasticTransform, \
                            RandomScale, Resize, Normalize, RandomCrop
from albumentations.pytorch import ToTensor

from utils.customDataset import threeChannelDataset
from utils.customWriter import customWriter
from utils.customModel import customResnet
import cv2

torch.autograd.set_detect_anomaly(True)
train_data = './data/training'
valid_data = './data/validation'
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
    model = customResnet(n_outputs=num_outputs)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', verbose=True)
    
    # Applies LogSoftmax internally
    criterion = nn.MSELoss().cuda()
    ce_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 57.9, 55.3, 55.4, 56.1, 48.8, 38.8,
                                                         29.4, 24.2, 24.9, 27.4, 28.9, 29.4, 30.3])).cuda()


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
            inputs = data['inputs'].to(device, dtype=torch.float32)
            # Augmentation adds channel axis
            targets = data['targets'].to(device, dtype=torch.float32)
            # Remove single channel 
            masks = torch.squeeze(data['masks'].to(device, dtype=torch.long))
            labels = data['class_labels']
            if epoch % 25==0 and idx==0:
                writer.plot_batch(f'Inputs at epoch {epoch}', inputs, targets, masks, labels, plot_target=True)

            optimizer.zero_grad()

           
            coords, heatmaps, unnorm_heatmaps, seg_pred = model(inputs)
            ce_loss = ce_criterion(seg_pred, masks)
            # Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, targets)
            # Per-location regularization losses
            reg_losses = dsntnn.js_reg_losses(heatmaps, targets, sigma_t=model.sigma)
            # Combine losses into an overall loss
            
            loss = dsntnn.average_loss(
                alpha*euc_losses+alpha*reg_losses )#, mask=data['present'].to(device, dtype=torch.float32))
            # Penalise large peak widths
            #loss += beta*torch.norm(model.sigma)**2
            
            loss += ce_loss
            
            writer.train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        
        #writer.add_histogram('Sigma', model.sigma, epoch)
        
        print('Train Loss:', np.mean(writer.train_loss))
        writer.add_scalar('training_loss', np.mean(writer.train_loss), epoch)
        

        loss_list = []
        
        with torch.set_grad_enabled(False):
            print('VALIDATION')
            for batch_idx, data in enumerate(valid_generator):
                inputs = data['inputs'].to(device, dtype=torch.float32)
                targets = data['targets'].to(device, dtype=torch.float32)
                labels = data['class_labels']
                masks = torch.squeeze(
                    data['masks'].to(device, dtype=torch.long))

                optimizer.zero_grad()

                coords, heatmaps, unnorm_heatmaps, seg_pred = model(inputs)
                if epoch % 25 == 0 and batch_idx == 0:
                    writer.plot_prediction(
                        f'Prediction at epoch {epoch}', inputs, coords, targets, plot_target=True)

                    writer.plot_segmentation(f'Segmentation at epoch {epoch}', inputs, seg_pred, masks, plot_target=False)
                    writer.plot_heatmap(f'Heatmaps at epoch {epoch}', inputs, heatmaps)
                    writer.plot_heatmap(
                        f'Unnormalized Heatmaps at epoch {epoch}', inputs, unnorm_heatmaps)
                
                ce_loss = ce_criterion(seg_pred, masks)
                # Per-location euclidean losses
                euc_losses = dsntnn.euclidean_losses(coords, targets)
                # Per-location regularization losses
                reg_losses = dsntnn.js_reg_losses(heatmaps, targets, sigma_t=model.sigma)
                # Combine losses into an overall loss
                valid_loss = dsntnn.average_loss(
                    alpha*euc_losses+alpha*reg_losses)#, mask=data['present'].to(device, dtype=torch.float32))
                #valid_loss += beta*torch.norm(model.sigma)**2
                valid_loss += ce_loss

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
    train_transforms = Compose([
        HorizontalFlip(p=0.5),
        Rotate(p=0.5, limit=5, border_mode=cv2.BORDER_CONSTANT, value=0),
        # ElasticTransform(p=0.5, approximate=True, alpha=75, sigma=8,
        #                  alpha_affine=50, border_mode=cv2.BORDER_CONSTANT, value=0),
        # RandomScale(),
        # Resize(512, 212),
        RandomCrop(height=470, width=452, p=0.5),
        Resize(626, 452),
        Normalize(mean=(0.485, 0.456, 0.406), std=(
            0.229, 0.224, 0.225), max_pixel_value=1),
        ToTensor()
    ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=True, label_fields=['class_labels']))

    valid_transforms = Compose([Normalize(mean=(0.485, 0.456, 0.406), std=(
        0.229, 0.224, 0.225), max_pixel_value=1), 
        ToTensor()], keypoint_params=A.KeypointParams(format='yx', remove_invisible=True, label_fields=['class_labels']))

    train_dataset = threeChannelDataset(train_data, transforms=train_transforms, normalise=False)
    valid_dataset = threeChannelDataset(valid_data, transforms=valid_transforms, normalise=False)

    # train_dataset.write_categories('overview.txt')
    # valid_dataset.write_categories('overview_valid.txt')

    train_generator = DataLoader(train_dataset, batch_size=batch_size)
    valid_generator = DataLoader(valid_dataset, batch_size=batch_size)

    model = train(train_generator, valid_generator)
    # Save model
    torch.save(model.state_dict(), output_path + 'model.pt')
    torch.cuda.empty_cache()
