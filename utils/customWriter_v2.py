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
        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']
        self.hist_colours = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'brown', 'pink', 'purple', 'k', 'gray', 'olive']


    def plot_inputs(self, title, img, targets=None):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Inputs')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            ax.imshow(plt_img)
            if targets is not None:
                for coords, vert in targets[idx]:
                    print(coords, vert)
                    y, x = coords
                    
                    ax.scatter(y, x)
                    ax.text(y, x, vert)
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
                'Input @ epoch: {} - idx: {}'.format(self.epoch, idx))
            ax.legend(loc='upper right')
        self.add_figure(title, fig)

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def reset_losses(self):
        self.train_loss, self.val_loss, self.class_loss = [], [], {
            n: [] for n in range(self.num_classes+1)}

    def reset_acc(self):
        self.train_acc, self.val_acc = [],  []


