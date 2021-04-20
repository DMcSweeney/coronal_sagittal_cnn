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



class customWriter(SummaryWriter):
    def __init__(self, log_dir, batch_size, num_classes, epoch=0):
        super().__init__()
        self.batch_size = batch_size
        self.epoch = epoch
        self.train_loss = []
        self.val_loss = []
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']
        self.hist_colours = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'brown', 'pink', 'purple', 'k', 'gray', 'olive']

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    @staticmethod
    def norm_img(img):
        return (img-img.min())/(img.max()-img.min())

    def reset_losses(self):
        self.train_loss, self.val_loss = [], []

    def reset_acc(self):
        self.train_acc, self.val_acc = [],  []

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
                for i in range(len(self.ordered_verts)):
                    if verts[idx, i] == 1:
                        y = coords[idx, i]
                        ax.axhline(y, c='y')
                        ax.text(0, y-5, self.ordered_verts[i], color='white')
            ax.set_title(
                'Input @ epoch: {} - idx: {}'.format(self.epoch, idx))
        self.add_figure(title, fig)

    def plot_histogram(self, title, histogram, labels=None, prediction=None):
        fig = plt.figure(figsize=(25, 15))
        plt.tight_layout()
        histogram = histogram.cpu().numpy()
        x = np.linspace(0, histogram.shape[-1], num=histogram.shape[-1])
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Histograms')
            for channel in range(histogram[idx].shape[0]):
                vert = self.ordered_verts[channel]
                if labels is not None:
                    if labels[idx, channel] == 0: continue
                if prediction is not None:
                    pred = prediction[idx, channel].cpu().numpy()
                    ax.plot(x, pred, '--', color=self.hist_colours[channel])
                data = histogram[idx, channel]
                ax.plot(x, data, label=vert, color=self.hist_colours[channel])
            ax.set_title(
                'Histogram @ epoch: {} - idx: {}'.format(self.epoch, idx))
            ax.legend(loc='upper right')
        self.add_figure(title, fig)

    def plot_prediction(self, title, img, prediction, targets=None, predicted=None):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        prediction = prediction.cpu().numpy()
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Inputs')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img)
            for i, coord in enumerate(prediction[idx]):
                if targets is not None:
                    _, verts = targets
                    if verts[idx, i] == 1:
                        ax.axhline(coord, c='w', linestyle='--')
                        ax.text(256, coord-5, self.ordered_verts[i], color='r')

            if targets is not None:
                coords, verts = targets
                for i in range(len(self.ordered_verts)):
                    if verts[idx, i] == 1:
                        y = coords[idx, i]
                        ax.axhline(y, c='y')
                        ax.text(0, y-5, self.ordered_verts[i], color='white')
            ax.set_title(
                'Prediction @ epoch: {} - idx: {}'.format(self.epoch, idx))
        self.add_figure(title, fig)




