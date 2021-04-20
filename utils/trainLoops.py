"""
Objects used for training, to clean-up
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import dsntnn

import torch
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam

from torchviz import make_dot
import graphviz
from torch.autograd import Variable



import utils.customModel_v2 as cm2
import utils.customWriter_v2 as cw2


class Locator():
    """
    ~Class for training vertebrae locator
    @params: 
      dir_name = directory name used for splitting tensorboard runs.   
    """
    def __init__(self, train_dataLoader, val_dataLoader, dir_name, device="cuda:0", 
                    batch_size=4, n_outputs=13, learning_rate=3e-3, num_epochs=200, output_path='./outputs/'):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.train_dataLoader = train_dataLoader
        self.val_dataLoader = val_dataLoader
        self.model = cm2.customUNet(n_outputs=n_outputs).cuda()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.L1Loss().cuda()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True)
        self.num_epochs = num_epochs
        self.writer = cw2.customWriter(log_dir=f'./runs/{dir_name}', batch_size=batch_size, num_classes=n_outputs)
        self.best_loss = 10000 # Initialise loss for saving best model
        self.output_path = output_path
    
    def forward(self):
        #~ Forward pass def
        for epoch in range(self.num_epochs+1):
            self.writer.epoch = epoch
            print(f'Epoch: {epoch}/{self.num_epochs}')
            #~TRAINING
            self.train(epoch=epoch, write2tensorboard=True, writer_interval=20)
            #~ VALIDATION
            self.validation(epoch=epoch, write2tensorboard=True, writer_interval=10)
            #* Save best model
            self.save_best_model()
        
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
            heatmaps = data['heatmap'].to(self.device, dtype=torch.float32)
            keypoints, labels = data['keypoints'].to(self.device, dtype=torch.float32), data['class_labels'].to(self.device, dtype=torch.int8)
            self.optimizer.zero_grad() #Reset gradients

            #! seg_out = output from segmentation head (B x N_OUTPUTS x H x W)
            #! heatmap = 1D heatmap (B x N_OUTPUTS x H x 1)
            #! coords = 1D coordinates (B x N_OUTPUTS x 1)

            pred_seg, pred_heatmap, pred_coords = self.model(sag_img, cor_img)
            l1_loss = dsntnn.l1_losses(pred_coords.unsqueeze(-1), keypoints.unsqueeze(-1))
            js_reg = dsntnn.js_reg_losses(
                heatmaps=pred_heatmap[..., 0], mu_t=keypoints.unsqueeze(-1), sigma_t=10)
            loss = dsntnn.average_loss(l1_loss+js_reg, mask=labels)
            self.writer.train_loss.append(loss.item())
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
                        f'Target heatmap at epoch {epoch}', heatmaps, labels=labels)

        print('Train Loss:', np.mean(self.writer.train_loss))
        self.writer.add_scalar('Training Loss', np.mean(
            self.writer.train_loss), epoch)
            
    def validation(self, epoch, write2tensorboard=True, writer_interval=10):
        #~Validation loop
        with torch.set_grad_enabled(False):
            print('Validation...')
            for idx, data in enumerate(tqdm(self.val_dataLoader)):
                #* Load data
                sag_img = data['sag_image'].to(self.device, dtype=torch.float32)
                cor_img = data['cor_image'].to(self.device, dtype=torch.float32)

                keypoints, labels = data['keypoints'].to(
                    self.device, dtype=torch.float32), data['class_labels'].to(self.device, dtype=torch.int8)
    
                pred_seg, pred_heatmap, pred_coords = self.model(
                    sag_img, cor_img)
                l1_loss = dsntnn.l1_losses(
                    pred_coords.unsqueeze(-1), keypoints.unsqueeze(-1))
                js_reg = dsntnn.js_reg_losses(
                    heatmaps=pred_heatmap[..., 0], mu_t=keypoints.unsqueeze(-1), sigma_t=10)

                loss = dsntnn.average_loss(l1_loss+js_reg, mask=labels)
                self.writer.val_loss.append(loss.item())

                if write2tensorboard:
                    #* Write predictions to tensorboard
                    if epoch % writer_interval == 0 and idx==0:
                        self.writer.plot_prediction(f'Prediction at epoch {epoch}', img=sag_img, prediction=pred_coords, targets=[keypoints, labels])
                        self.writer.plot_histogram(f'Predicted Heatmap at epoch {epoch}', pred_heatmap[..., 0], labels=labels)
            print('Validation Loss:', np.mean(self.writer.val_loss))
            self.scheduler.step(np.mean(self.writer.val_loss))
            self.writer.add_scalar('Validation Loss', np.mean(self.writer.val_loss), epoch)

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

    def write2file(self, array, output_path='./logs/heatmaps/'):
        #~ Write arrays to file for sanity checking 
        print('Writing to file')
        arr = array.cpu().detach().numpy()
        heatmap = np.tile(arr[..., np.newaxis], reps=(1, 1, 256))
        for i in range(heatmap.shape[1]):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(heatmap[0, i])
            fig.savefig(output_path+f'heatmap_{i}.png')

    def save_best_model(self):
        #~ Check if latest validation is min. If True, save.
        loss1 = np.mean(self.writer.val_loss)
        is_best = loss1 < self.best_loss
        self.best_loss = min(loss1, self.best_loss)
        if is_best:
            print('Saving best model')
            torch.save(self.model.state_dict(),
            self.output_path + 'best_model.pt')
        
