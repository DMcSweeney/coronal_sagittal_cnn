"""
Custom tensorboard writer class
"""
from random import randrange
import matplotlib
matplotlib.use('agg')

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from scipy.special import softmax



class customWriter(SummaryWriter):
    def __init__(self, log_dir, batch_size, epoch, num_classes):
        super(customWriter, self).__init__()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_classes = num_classes
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.class_loss = {n: [] for n in range(num_classes+1)}
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def reset_losses(self):
        self.train_loss, self.val_loss, self.class_loss = [], [], {
            n: [] for n in range(self.num_classes+1)}


    def reset_acc(self):
        self.train_acc, self.val_acc = [],  []

    def plot_batch(self, tag, img, masks, plot_target=False):
        """
        Plot batches in grid

        Args: tag = identifier for plot (string)
              images = input batch (torch.tensor)
              targets = ground truth normalised coordinates (shape: [batch, n_locations, 2])
        """
        fig = plt.figure(figsize=(24, 24))
        
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, xticks=[], yticks=[], label='Inputs')
            ax.imshow(img[idx, 0].cpu().numpy())
            if plot_target:
                ax.imshow(masks[idx].cpu().numpy(), alpha=0.5)
            ax.set_title(
                'Input @ epoch: {} - idx: {}'.format(self.epoch, idx))
        self.add_figure(tag, fig)

    @staticmethod
    def get_com_from_segmentation():
        bin_pred = np.where(proj_pred == channel, 1, 0)
        y_mask, x_mask = bin_pred*y_temp, bin_pred*x_temp
        y = y_mask[y_mask != 0].mean()
        x = x_mask[x_mask != 0].mean()
        if np.isnan(y):
            y = 0
        if np.isnan(x):
            x = 0
        com = (y, x)
        return com

    def plot_prediction(self, tag, img, prediction, target, plot_target=True):
        """
        Plot predictions vs target segmentation.
        Args: tag = identifier for plot (string)
                img = CT slice
              prediction = batch output of trained model (torch.tensor)
              target = batch ground-truth segmentations (torch.tensor)
              masks = segmentations
        """
    
        fig = plt.figure(figsize=(24, 24))
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, xticks=[], yticks=[], label='predictions')
            ax.imshow(img[idx, 0].cpu().numpy(), cmap='gray')
            y_shape, x_shape = img[idx, 0].shape[0]/2, img[idx, 0].shape[1]/2
            for i, vert in enumerate(self.ordered_verts):
                if vert == 'L3':
                    coords = prediction[idx, 0].cpu().numpy()
                    out_x, out_y = coords[1]+1, coords[0]+1
                    plt_coords = [out_x*x_shape, out_y*y_shape]
                    ax.scatter(plt_coords[0], plt_coords[1], marker='+')
                    ax.text(plt_coords[0]+5, plt_coords[1], vert,
                                    color='red', fontsize=15, alpha=1)
            if plot_target:        
                for i, vert in enumerate(self.ordered_verts):
                    coords= target[idx, i].cpu().numpy()
                    out_x, out_y = coords[1]+1, coords[0]+1
                    plt_coords = [out_x*x_shape, out_y*y_shape]
                    ax.scatter(plt_coords[0], plt_coords[1], marker='o')
                    ax.text(plt_coords[0]+5, plt_coords[1], vert,
                  color = 'white', fontsize = 15, alpha=1)

            ax.set_title(
                'prediction @ epoch: {} - idx: {}'.format(self.epoch, idx))
        self.add_figure(tag, fig)

    def plot_segmentation(self, tag, img, segmentation, mask, plot_target=True):
        fig = plt.figure(figsize=(24, 24))
        #segmentation = np.log(softmax(segmentation.cpu().numpy(), axis=1)) 
        segmentation  = np.round(self.sigmoid(segmentation).cpu().numpy())
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, xticks=[], yticks=[], label='predictions')
            ax.imshow(img[idx, 0].cpu().numpy(), cmap='gray')
            seg_proj = segmentation[idx, 0]
            print(seg_proj.max(), seg_proj.min())
            ax.imshow(seg_proj, alpha=0.5, cmap='Oranges')
            if plot_target:
                ax.imshow(mask[idx].cpu().numpy(), alpha=0.5, cmap='Blues')

            ax.set_title('Segmentation @ epoch: {} - idx: {}'.format(self.epoch, idx))
        self.add_figure(tag, fig)

    def plot_heatmap(self, tag, img, heatmap):
        fig = plt.figure(figsize=(24, 24))
        for idx in np.arange(self.batch_size):
            channel = 0
            vert= 'T12'
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, xticks=[], yticks=[], label='predictions')
            ax.imshow(img[idx, 0].cpu().numpy(), cmap='gray')
            heat_proj = heatmap[idx, channel].cpu().numpy()
            ax.imshow(heat_proj, alpha=0.5)

            ax.set_title('Heatmap @ epoch: {} - idx: {} - channel: {} - vert: {} '.format(self.epoch, idx, channel, vert))
        self.add_figure(tag, fig)

    def plot_histogram(self, tag, prediction):
        print('Plotting histogram')
        fig = plt.figure(figsize=(24, 24))
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, yticks=[], label='histogram')
            pred_norm = (prediction[idx, 0]-prediction[idx, 0].min())/(
                prediction[idx, 0].max()-prediction[idx, 0].min())
            ax.hist(pred_norm.cpu().flatten(), bins=100)
            ax.set_title(
                f'Prediction histogram @ epoch: {self.epoch} - idx: {idx}')
        self.add_figure(tag, fig)

    def per_class_loss(self, prediction, target, criterion, alpha=None):
        # Predict shape: (4, 1, 512, 512)
        # Target shape: (4, 1, 512, 512)
        #pred, target = prediction.cpu().numpy(), target.cpu().numpy()
        pred, target = prediction, target
        for class_ in range(self.num_classes + 1):
            class_pred, class_tgt = torch.where(
                target == class_, pred, torch.tensor([0], dtype=torch.float32).cuda()),  torch.where(target == class_, target, torch.tensor([0], dtype=torch.float32).cuda())

            #class_pred, class_tgt = pred[target == class_], target[target == class_]
            if alpha is not None:
                loss = criterion(class_pred, class_tgt, alpha)
                #bce_loss, dice_loss = criterion(class_pred, class_tgt, alpha)
            else:
                loss = criterion(class_pred, class_tgt)
                #bce_loss, dice_loss = criterion(class_pred, class_tgt)
            #loss = bce_loss + dice_loss
            self.class_loss[class_].append(loss.item())

    def write_class_loss(self):
        for class_ in range(self.num_classes+1):
            self.add_scalar(f'Per Class loss for class {class_}', np.mean(
                self.class_loss[class_]), self.epoch)
