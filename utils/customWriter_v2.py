"""
Custom tensorboard writer class
"""
import matplotlib
matplotlib.use('agg')
from scipy.special import softmax
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
from random import randrange
import seaborn as sns
import dsntnn


class customWriter(SummaryWriter):
    def __init__(self, log_dir, batch_size, num_classes, epoch=0):
        super().__init__()
        self.batch_size = batch_size
        self.epoch = epoch
        self.losses = {}
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']
        self.hist_colours = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 
        'brown', 'pink', 'purple', 'k', 'gray', 'olive']

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    @staticmethod
    def norm_img(img):
        if img.shape[-1] == 1:
            return (img-img.min())/(img.max()-img.min())
        elif img.shape[-1] == 3:
            norm_img = []
            for chan in np.arange(0, 3):
                norm_img.append(
                    (img[..., chan]-img[..., chan].min())/(img[..., chan].max()-img[..., chan].min()))
            return np.stack(norm_img, axis=-1)

    def init_losses(self, keys):
        for key in keys:
            self.losses[key] = []
            
    def reset_losses(self):
        for loss in self.losses.keys():
            self.losses[loss] = []

    def plot_inputs(self, title, img, targets=None):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Inputs')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img)
            if targets is not None:
                coords, verts = targets
                coords = coords.cpu()
                coords = dsntnn.normalized_to_pixel_coordinates(
                    coords[idx], size=plt_img.shape[0])
                for i in range(len(self.ordered_verts)):
                    if verts[idx, i] == 1:
                        y = coords[i]
                        ax.axhline(y, c='y')
                        ax.text(0, y-5, self.ordered_verts[i], color='white')
            ax.set_title(
                'Input @ epoch: {} - idx: {}'.format(self.epoch, idx))
        self.add_figure(title, fig)

    def plot_histogram(self, title, histogram, targets=None, detect=False):
        fig = plt.figure(figsize=(25, 15))
        plt.tight_layout()
        histogram = histogram.cpu().numpy()
        x = np.linspace(0, histogram.shape[-1], num=histogram.shape[-1])
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Histograms')
            if not detect:
                for channel in range(histogram[idx].shape[0]):
                    vert = self.ordered_verts[channel]
                    if targets is not None:
                        #* Plot target heatmaps
                        heatmaps, labels = targets
                        if heatmaps is not None and labels[idx, channel] == 1:
                            tgt = heatmaps[idx, channel].cpu().numpy()
                            ax.plot(x, tgt, '-', color=self.hist_colours[channel])
                        if labels[idx, channel] == 1:
                            #* Only plot visible levels
                            data = histogram[idx, channel]
                            ax.plot(x, data, '--', label=vert, color=self.hist_colours[channel])
            else:
                data = histogram[idx, 0]
                ax.plot(x, data, '-')
            
            ax.set_title(
                'Histogram @ epoch: {} - idx: {}'.format(self.epoch, idx))
            ax.legend(loc='upper right')
        self.add_figure(title, fig)

    def plot_mask(self, title, img, prediction, apply_sigmoid=True):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        if apply_sigmoid:
            mask = self.sigmoid(prediction).cpu().numpy()
        else:
            mask = prediction.cpu().numpy()
        
        mask = np.where(mask == 0, np.nan, mask)
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2, idx+1, label='Inputs')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img)  
            ax.imshow(mask[idx, 0], alpha=0.5)
            ax.set_title(
                    'Predicted Mask @ epoch: {} - idx: {}'.format(self.epoch, idx))
        self.add_figure(title, fig, global_step=self.epoch)

    def plot_prediction(self, title, img, prediction, targets=None, predicted=None):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        prediction = prediction.cpu().numpy()
        
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Inputs')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            norm_prediction = dsntnn.normalized_to_pixel_coordinates(
                prediction[idx], size=plt_img.shape[0])
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img)
            for i, coord in enumerate(norm_prediction):
                if targets is not None:
                    _, verts = targets
                    if verts[idx, i] == 1:
                        ax.axhline(coord, c='w', linestyle='--')
                        ax.text(512, coord-5, self.ordered_verts[i], color='r')

            if targets is not None:
                coords, verts = targets
                coords = coords.cpu()
                coords = dsntnn.normalized_to_pixel_coordinates(
                    coords[idx], size=plt_img.shape[0])
                for i in range(len(self.ordered_verts)):
                    if verts[idx, i] == 1:
                        y = coords[i]
                        ax.axhline(y, c='y')
                        ax.text(0, y-5, self.ordered_verts[i], color='white')
            ax.set_title(
                'Prediction @ epoch: {} - idx: {}'.format(self.epoch, idx))
        self.add_figure(title, fig)




