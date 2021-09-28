"""
Custom tensorboard writer class
"""
import matplotlib
from numpy.lib.shape_base import dstack
from tensorboard.compat.proto.summary_pb2 import DATA_CLASS_TENSOR
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
from einops import rearrange
import torch.nn.functional as F

class customWriter(SummaryWriter):
    def __init__(self, log_dir, batch_size, num_classes, epoch=0):
        super().__init__()
        self.batch_size = batch_size
        self.epoch = epoch
        self.losses = {}
        # self.cmap = sns.cubehelix_palette(
        #     start=1.5, rot=-1.5, gamma=0.8, as_cmap=True)
        self.cmap = 'viridis'
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

    @staticmethod
    def sharpen_heatmap(heatmap, alpha=2):
        sharp_map = heatmap ** alpha
        eps = 1e-24
        flat_map = sharp_map.flatten(2).sum(-1)[..., None, None]
        flat_map += eps
        return sharp_map/flat_map

    def init_losses(self, keys):
        for key in keys:
            self.losses[key] = []
            
    def reset_losses(self):
        for loss in self.losses.keys():
            self.losses[loss] = []

    def plot_heatmap(self, title, img, heatmap, apply_softmax=True, norm_coords=False, labels=None):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        heatmap = self.sharpen_heatmap(heatmap, alpha=2)
        if apply_softmax:
            heatmap = dsntnn.flat_softmax(heatmap).cpu().numpy()
        else:
            heatmap = heatmap.cpu().numpy()
        img = rearrange(img, 'b c h w -> b h w c')
        cmap=sns.cubehelix_palette(start=1.5, rot=-1, hue=1.4, gamma=1.0, as_cmap=True)
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2,
                                 self.batch_size // 2, idx+1, label='Inputs')
            plt_img = self.norm_img(img[idx].cpu().numpy())
            ax.imshow(plt_img, cmap='gray')
            coords = dsntnn.dsnt(
                torch.tensor(heatmap), normalized_coordinates=norm_coords)
            if norm_coords:
                coords = dsntnn.normalized_to_pixel_coordinates(coords, size=(512, 512))

            for channel in range(coords.shape[1]):
                x, y = coords[idx, channel]
                if channel == 0:
                    continue
                if labels is not None and labels[idx, channel-1] == 1:
                    vert = self.ordered_verts[channel - 1]

                    ax.scatter(x, y, s=20, c='r', marker='+')
                    ax.text(x, y, vert, c='y', size=15)
            plt_heatmap = np.max(heatmap[idx], axis=0)
            #plt_heatmap = np.where(plt_heatmap == 0, np.nan, plt_heatmap)
            #ax.imshow(plt_heatmap, alpha=0.5, cmap=self.cmap)
            ax.imshow(plt_heatmap, cmap=cmap, alpha=0.5)
            ax.set_title(title)
        #fig.savefig('../outputs/heatmap_test.png')
        self.add_figure(title, fig, global_step=self.epoch)

    def plot_mask(self, title, img, prediction, apply_sigmoid=False):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        if apply_sigmoid:
            mask = self.sigmoid(prediction).cpu().numpy()
        else:
            mask = prediction.cpu().numpy()
        img = rearrange(img, 'b c h w -> b h w c')
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2, idx+1, label='Inputs')
            plt_img = self.norm_img(img[idx].cpu().numpy())
            ax.imshow(plt_img)
            ax.imshow(mask[idx], alpha=0.5, cmap=self.cmap)
            ax.set_title(title)
        self.add_figure(title, fig, global_step=self.epoch)

    def plot_prediction(self, title, img, prediction, type_, ground_truth=None, apply_norm=False,
                coords=None, gt_coords=None, labels=None):
        #* Type == ['mask', 'heatmap']
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0)

        pred_coords = dsntnn.dsnt(prediction, normalized_coordinates=False) if type_ == 'heatmap' else None

        if apply_norm:
            if type_ == 'mask':
                #! Only for binary masks
                prediction = F.softmax(prediction, dim=1).cpu().numpy()
                #prediction = self.sigmoid(prediction).cpu().numpy()
                
            elif type_ == 'heatmap':
                prediction = dsntnn.flat_softmax(prediction).cpu().numpy()
                if labels is not None:
                    labels = self.sigmoid(labels).cpu().numpy()
                    labels = np.where(labels > 0.5, 1, 0)
                    print(labels)
        else:
            prediction = prediction.cpu().numpy()

        if labels is not None:
            labels = self.sigmoid(labels).cpu().numpy()
            labels = np.where(labels > 0.5, 1, 0)
            print(labels)
            
        img = rearrange(img, 'b c h w -> b h w c')
        idx = 0
        plt_img = self.norm_img(img[idx].cpu().numpy())
        ax[0].axis('off')
        ax[1].axis('off')
        ax[0].imshow(plt_img[..., 1], cmap='gray')
        ax[0].set_title('Ground-truth')
        ax[1].imshow(plt_img[..., 1], cmap='gray')
        ax[1].set_title('Prediction')
        if type_ == 'heatmap':
            if ground_truth is not None:
                ax[0].imshow(
                    np.max(ground_truth[idx].cpu().numpy(), axis=0), alpha=0.5, cmap=self.cmap)
            ax[1].imshow(np.max(prediction[idx], axis=0), alpha=0.5, cmap=self.cmap)
            #* Plot coordinates according to heatmap
            if pred_coords is not None:
                for i, vert in enumerate(self.ordered_verts):
                    if labels is not None and labels[idx, i] == 0:
                        continue
                    x, y = pred_coords[idx, i].cpu().numpy()[::-1]
                    ax[1].scatter(x, y, marker='^', c='g', s=15)
                    ax[1].text(x, y, vert, c='g', size=15)

        elif type_ == 'mask':
            if ground_truth is not None:
                gt = np.where(ground_truth[idx].cpu().numpy() == 0, np.nan, ground_truth[idx].cpu().numpy())
                ax[0].imshow(gt, alpha=0.5, cmap=self.cmap)
            pred = np.where(np.argmax(prediction[idx], axis=0) == 0, np.nan, np.argmax(
                prediction[idx], axis=0))
            ax[1].imshow(pred, alpha=0.5, cmap=self.cmap)
        if coords is not None:
            for i, vert in enumerate(self.ordered_verts):
                if labels is not None and labels[idx, i] == 0: continue
                x, y = coords[idx, i].cpu().numpy()[::-1]
                ax[1].scatter(x, y, marker='+', c='r', s=15)
                ax[1].text(x, y, vert, c='r', size=15)
        if gt_coords is not None:
            for i, vert in enumerate(self.ordered_verts):
                x, y = gt_coords[idx, i].cpu().numpy()[::-1]
                ax[0].scatter(x, y, marker='+', c='r', s=15)
                ax[0].text(x, y, vert, c='r', size=15)

        self.add_figure(title, fig, global_step=self.epoch)



