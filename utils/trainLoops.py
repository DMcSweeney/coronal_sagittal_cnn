"""
Objects used for training, to clean-up
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from tqdm import tqdm
import dsntnn

import torch
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam

from torchviz import make_dot
import graphviz
from torch.autograd import Variable

import seaborn as sns

import utils.customModel_v2 as cm2
import utils.customWriter_v2 as cw2
import utils.customLosses as cl

class Locator():
    """
    ~Class for training vertebrae locator
    @params: 
      dir_name = directory name used for splitting tensorboard runs.   
    """
    def __init__(self, train_dataLoader, val_dataLoader, test_dataLoader, dir_name, device="cuda:0", 
                    batch_size=4, n_outputs=13, learning_rate=3e-3, num_epochs=200, output_path='./outputs/'):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.train_dataLoader = train_dataLoader
        self.val_dataLoader = val_dataLoader
        self.test_dataLoader = test_dataLoader
        self.model = cm2.customUNet(n_outputs=n_outputs).cuda()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.L1Loss().cuda()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True)
        self.num_epochs = num_epochs
        self.writer = cw2.customWriter(log_dir=f'./runs/{dir_name}', batch_size=batch_size, num_classes=n_outputs)
        self.best_loss = 10000 # Initialise loss for saving best model
        self.output_path = output_path
    
    def forward(self, num_epochs=200, model_name='best_model.pt'):

        #~ Forward pass def
        for epoch in range(num_epochs+1):
            self.writer.epoch = epoch
            print(f'Epoch: {epoch}/{num_epochs}')
            #~TRAINING
            self.train(epoch=epoch, write2tensorboard=True, writer_interval=20)
            #~ VALIDATION
            self.validation(epoch=epoch, write2tensorboard=True, writer_interval=10)
            #* Save best model
            self.save_best_model(model_name=model_name)
        
    def train(self, epoch, write2tensorboard=True, writer_interval=20, viz_model=False):
        #~Main training loop
        #@param:
        #    writer_interval = write to tensorboard every x epochs

        #* Allow param optimisation & reset losses
        self.model.train()
        self.writer.reset_losses()
        #*Visualise model using graphviz
        if viz_model:
            self.viz_model(output_path='./logs/')
        # ** Training Loop **
        for idx, data in enumerate(tqdm(self.train_dataLoader)):
            #*Load data
            sag_img = data['sag_image'].to(self.device, dtype=torch.float32)
            cor_img = data['cor_image'].to(self.device, dtype=torch.float32)
            heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
            keypoints, labels = data['keypoints'].to(self.device, dtype=torch.float32), data['class_labels'].to(self.device, dtype=torch.int8)
            self.optimizer.zero_grad() #Reset gradients
            
            #! seg_out = output from segmentation head (B x N_OUTPUTS x H x W)
            #! heatmap = 1D heatmap (B x N_OUTPUTS x H x 1)
            #! coords = 1D coordinates (B x N_OUTPUTS x 1)

            pred_seg, pred_heatmap, pred_coords = self.model(sag_img, cor_img)
            #* Loss + Regularisation

            l1_loss = torch.nn.functional.mse_loss(
                pred_coords, keypoints, reduction='none')
            js_reg = cl.js_reg(pred_heatmap[..., 0], heatmap)
            loss = dsntnn.average_loss(l1_loss+js_reg, mask=labels)
            self.writer.train_loss.append(loss.item())
            #* Optimiser step
            loss.backward()
            self.optimizer.step()

            if write2tensorboard:
                # ** Write inputs to tensorboard
                if epoch % writer_interval ==0  and idx == 0:
                    self.writer.plot_inputs(f'Sagittal Inputs at epoch {epoch}', sag_img, targets=[
                                       keypoints, labels])
                    self.writer.plot_inputs(
                        f'Coronal Inputs at epoch {epoch}', cor_img)
                    self.writer.plot_histogram(
                        f'Target heatmap at epoch {epoch}', heatmap, targets=[None, labels])
            
        print('Train Loss:', np.mean(self.writer.train_loss))
        self.writer.add_scalar('Training Loss', np.mean(
            self.writer.train_loss), epoch)
            
    def validation(self, epoch, write2tensorboard=True, writer_interval=10, write_gif=True):
        #~Validation loop
        with torch.set_grad_enabled(False):
            print('Validation...')
            for idx, data in enumerate(tqdm(self.val_dataLoader)):
                #* Load data
                sag_img = data['sag_image'].to(self.device, dtype=torch.float32)
                cor_img = data['cor_image'].to(self.device, dtype=torch.float32)
                heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
                keypoints, labels = data['keypoints'].to(
                    self.device, dtype=torch.float32), data['class_labels'].to(self.device, dtype=torch.int8)
                pred_seg, pred_heatmap, pred_coords = self.model(
                    sag_img, cor_img)
                #* Loss + Regularisation
                l1_loss = torch.nn.functional.mse_loss(
                    pred_coords, keypoints, reduction='none')
                js_reg = cl.js_reg(pred_heatmap[..., 0], heatmap)
                loss = dsntnn.average_loss(l1_loss+js_reg, mask=labels)
                self.writer.val_loss.append(loss.item())
                self.writer.reg.append(js_reg[labels == 1].mean().item())
                self.writer.l1.append(l1_loss[labels == 1].mean().item())
                if write_gif:
                    if idx == 0:
                        self.write2file(
                            pred_heatmap[0, ..., 0], heatmap[0], epoch=epoch)
                if write2tensorboard:
                    #* Write predictions to tensorboard
                    if epoch % writer_interval == 0 and idx==0:
                        self.writer.plot_prediction(f'Prediction at epoch {epoch}', img=sag_img, prediction=pred_coords, targets=[keypoints, labels])
                        self.writer.plot_histogram(f'Predicted Heatmap at epoch {epoch}', pred_heatmap[..., 0], targets=[heatmap, labels])
            print('Validation Loss:', np.mean(self.writer.val_loss))
            self.scheduler.step(np.mean(self.writer.val_loss))
            self.writer.add_scalar('Validation Loss', np.mean(self.writer.val_loss), epoch)
            self.writer.add_scalar('Regularisation', np.mean(self.writer.reg), epoch)
            self.writer.add_scalar('L1-Loss', np.mean(self.writer.l1), epoch)

    def inference(self, model_name='best_model.pt', plot_output=False):
        #~ Model Inference
        print('Inference...')
        self.model.load_state_dict(torch.load(self.output_path + model_name))
        self.model.eval()
        all_ids = []
        all_pred_coords = []
        all_pred_heatmaps = []
        all_labels = []

        with torch.set_grad_enabled(False):
            for idx, data in enumerate(tqdm(self.test_dataLoader)):
                #* Load data
                sag_img = data['sag_image'].to(
                    self.device, dtype=torch.float32)
                cor_img = data['cor_image'].to(self.device, dtype=torch.float32)
                heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
                keypoints, labels = data['keypoints'].to(
                    self.device, dtype=torch.float32), data['class_labels'].to(self.device, dtype=torch.int8)
                ids = data['id']

                #* Get predictions
                pred_seg, pred_heatmap, pred_coords = self.model(sag_img, cor_img)
               
                if plot_output:
                    #* Plot predictions
                    #todo Add histograms + data for analysis
                    self.plot_predictions(ids, sag_img, pred_coords, targets=[keypoints, labels])
                    self.plot_distribution(ids, pred_heatmap[..., 0], targets=[heatmap, labels])
                    self.plot_heatmap(ids, pred_heatmap[..., 0], heatmap)
                


                all_ids.append(ids)
                all_pred_coords.append(pred_coords.cpu().numpy()))
                all_pred_heatmaps.append(pred_heatmap.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_ids = np.concatenate(all_ids, axis=0)
        all_pred_coords = np.concatenate(all_pred_coords, axis=0)
        all_pred_heatmaps = np.concatenate(all_pred_heatmaps, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        print(all_ids.shape, all_pred_heatmaps.shape, all_pred_coords.shape, all_labels.shape)
        #** Save predictions to npz file for post-processing
        print('SAVING PREDICTIONS...')
        np.savez(self.output_path + 'predictions.npz', ids=all_ids,
                    coords=all_pred_coords, heatmaps=all_pred_heatmaps, labels=all_labels)

    def viz_model(self, output_path='./logs/'):
        #~ View model architecture
        print('Vis model')
        input_shape = (4, 3, 512, 256) 
        model = self.model
        #* Create placeholder
        x = Variable(torch.randn(input_shape)).to('cuda')
        seg_out, heatmap, coords = model(x, x)
        #* Only follow path of coordinates
        graph = make_dot(coords.mean(), params=dict(model.named_parameters()))
        graph.render(output_path + 'graph.png')
        #summary(model, input_shape)

    def write2file(self, array, targets, epoch,output_path='./logs/gifs/'):
        #~ Write arrays to file for sanity checking 
        #* Colormaps for heatmaps
        plt.style.use('dark_background')
        cmap = sns.cubehelix_palette(
            start=0.5, rot=-1., hue=1, gamma=1, as_cmap=True)
        arr = array.cpu().detach().numpy()
        targets = targets.cpu().numpy()
        dist = zoom(input=arr, zoom=(10, 1), order=1)
        tgt = zoom(input=targets, zoom=(10, 1), order=1)
        fig, ax = plt.subplots(1, 2, figsize=(5, 10))
        ax[0].set_title(f'Epoch: {epoch}')
        ax[1].set_title('Ground-truth')
        ax[0].imshow(dist.T, cmap=cmap)
        ax[1].imshow(tgt.T, cmap=cmap)
        fig.savefig(output_path + f'no_scheduler_heatmap_epoch_{epoch}.png')    
        plt.close() 

    def save_best_model(self, model_name='best_model.pt'):
        #~ Check if latest validation is min. If True, save.
        loss1 = np.mean(self.writer.val_loss)
        is_best = loss1 < self.best_loss
        self.best_loss = min(loss1, self.best_loss)
        if is_best:
            print('Saving best model')
            torch.save(self.model.state_dict(),
            self.output_path + model_name)
    
    def plot_predictions(self, names, img, pred, targets=None):
        plt.style.use('dark_background')
        #~Write model predictions to files
        fig = plt.figure(figsize=(10,10))
        plt.tight_layout()
        prediction = pred.cpu().numpy()
        for idx in np.arange(len(names)):
            plt.axis('off')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            norm_prediction = dsntnn.normalized_to_pixel_coordinates(
                prediction[idx], size=plt_img.shape[0])
            plt_img = self.writer.norm_img(plt_img)
            plt.imshow(plt_img)
            for i, coord in enumerate(norm_prediction):
                if targets is not None:
                    _, verts = targets
                    if verts[idx, i] == 1:
                        plt.axhline(coord, c='w', linestyle='--', linewidth=2)
                        plt.text(512, coord-5, self.writer.ordered_verts[i], color='r')
            if targets is not None:
                coords, verts = targets
                coords = coords.cpu()
                coords = dsntnn.normalized_to_pixel_coordinates(
                    coords[idx], size=plt_img.shape[0])
                for i in range(len(self.writer.ordered_verts)):
                    if verts[idx, i] == 1:
                        y = coords[i]
                        plt.axhline(y, c='y', linewidth=2)
                        plt.text(0, y-5, self.writer.ordered_verts[i], color='white')
            fig.savefig(self.output_path + f'images/{names[idx]}.png')
            plt.clf()
        plt.close()

    def plot_distribution(self, names, pred, targets=None):
        #~ Plot distributiions to file
        plt.style.use('dark_background')
        fig =  plt.figure(figsize=(10, 7.5))
        plt.tight_layout()
        histogram = pred.cpu().numpy()
        x = np.linspace(0, histogram.shape[-1], num=histogram.shape[-1])
        for idx in np.arange(len(names)):
            for channel in range(histogram[idx].shape[0]):
                vert = self.writer.ordered_verts[channel]
                if targets is not None:
                    #* Plot target heatmaps
                    heatmaps, labels = targets
                    if heatmaps is not None and labels[idx, channel] == 1:
                        tgt = heatmaps[idx, channel].cpu().numpy()
                        plt.plot(x, tgt, '-', color=self.writer.hist_colours[channel])
                    if labels[idx, channel] == 1:
                        #* Only plot visible levels
                        data = histogram[idx, channel]
                        plt.plot(x, data, '--', label=vert,
                                color=self.writer.hist_colours[channel])
            plt.legend(loc='upper right')
            fig.savefig(self.output_path + f'distributions/{names[idx]}.png')
            plt.clf()
        plt.close()

    def plot_heatmap(self, names, pred, target):
        #~ Write arrays to file for sanity checking
        #* Colormaps for heatmaps
        plt.style.use('dark_background')
        cmap = sns.cubehelix_palette(
            start=0.5, rot=-1., hue=1, gamma=1, as_cmap=True)
        fig, ax = plt.subplots(1, 2, figsize=(5, 10))   
        for idx in np.arange(len(names)):
            arr = pred[idx].cpu().numpy()
            targets = target[idx].cpu().numpy()
            tgt = zoom(input=targets, zoom=(10, 1), order=1)
            dist = zoom(input=arr, zoom=(10, 1), order=1)

            ax[0].set_title('Prediction')
            ax[1].set_title('Ground-truth')
            ax[0].imshow(dist.T, cmap=cmap)
            ax[1].imshow(tgt.T, cmap=cmap)
            fig.savefig(self.output_path + f'heatmaps/{names[idx]}.png')
            plt.clf()
        plt.close()
