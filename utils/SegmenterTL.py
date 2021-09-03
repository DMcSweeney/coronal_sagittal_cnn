"""
Training loops for detection model

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
import torch.optim.swa_utils as swa

from torchviz import make_dot
import graphviz
from torch.autograd import Variable

import seaborn as sns

import utils.customModel_v2 as cm2
import utils.customWriter_v2 as cw2


class Segmenter():
    """
    ~Class for training vertebrae detector/segmentation model
    @params: 
      dir_name = directory name used for splitting tensorboard runs.   
    """
    def __init__(self, train_dataLoader, val_dataLoader, test_dataLoader, dir_name, device="cuda:0", 
                    batch_size=4, n_outputs=13, learning_rate=3e-3, num_epochs=200, output_path='./outputs/', model_path=None, SWA=False):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.train_dataLoader = train_dataLoader
        self.val_dataLoader = val_dataLoader
        self.test_dataLoader = test_dataLoader
        self.model = cm2.customUNet(n_outputs=n_outputs, classifier=False).cuda()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.criterion = nn.BCEWithLogitsLoss().cuda()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True)
        self.num_epochs = num_epochs
        self.writer = cw2.customWriter(log_dir=f'./runs/{dir_name}', batch_size=batch_size, num_classes=n_outputs)
        self.best_loss = 10000 # Initialise loss for saving best model
        self.output_path = output_path
        self.model_path = model_path

        #** Stochastic Weight Averaging (https://pytorch.org/docs/stable/optim.html#putting-it-all-together)
        self.swa = SWA
        self.swa_model = swa.AveragedModel(self.model) if SWA else None
        self.swa_scheduler = swa.SWALR(
            self.optimizer, swa_lr=0.05) if SWA else None  # LR set to large value
        self.swa_start = 100 if SWA else None  # START EPOCH from SWA
    
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def norm_img(img):
        return (img-img.min())/(img.max()-img.min())

    def forward(self, num_epochs=200, model_name='best_model.pt'):
        #~ Forward pass def
        for epoch in range(num_epochs+1):
            self.writer.epoch = epoch
            print(f'Epoch: {epoch}/{num_epochs}')
            #~TRAINING
            self.train(epoch=epoch, write2tensorboard=True, writer_interval=20)
            #~ VALIDATION
            self.validation(epoch=epoch, write2tensorboard=True, writer_interval=10, write_gif=False)
            #* Save best model
            self.save_best_model(model_name=model_name)
        
        #* Update batch norm stats. for SWA model
        if self.swa:
            print('Updating batch norm stats')
            for data in tqdm(self.train_dataLoader):
                img = data['sag_image'].to(self.device, dtype=torch.float32)
                self.swa_model(img)
            print('Saving best model')
            torch.save(self.swa_model.state_dict(),
                       self.output_path + f"{model_name.split('.')[0]}_SWA.pt")


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
            img = data['sag_image'].to(self.device, dtype=torch.float32)
            mask = data['mask'].to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad() #Reset gradients
        
            pred_seg = self.model(img)
            #* Loss + Regularisation
            loss = self.criterion(pred_seg, mask)
            self.writer.train_loss.append(loss.item())
            #* Optimiser step
            loss.backward()
            self.optimizer.step()
            if write2tensorboard:
                # ** Write inputs to tensorboard
                if epoch % writer_interval ==0  and idx == 0:
                    self.writer.plot_mask(
                        f'Ground-truth', img=img, prediction=mask)
            
        print('Train Loss:', np.mean(self.writer.train_loss))
        self.writer.add_scalar('Training Loss', np.mean(
            self.writer.train_loss), epoch)
            
    def validation(self, epoch, write2tensorboard=True, writer_interval=10, write_gif=False):
        #~Validation loop
        with torch.set_grad_enabled(False):
            print('Validation...')
            for idx, data in enumerate(tqdm(self.val_dataLoader)):
                #* Load data
                img = data['sag_image'].to(self.device, dtype=torch.float32)
                mask = data['mask'].to(self.device, dtype=torch.float32)
                pred_seg= self.model(img)
                
                #* Loss 
                val_loss = self.criterion(pred_seg, mask)

                self.writer.val_loss.append(val_loss.item())
                
                if write2tensorboard:
                    #* Write predictions to tensorboard
                    if epoch % writer_interval == 0 and idx==0:
                        self.writer.plot_mask(
                            f'Predicted mask', img=img, prediction=pred_seg)
            print('Validation Loss:', np.mean(self.writer.val_loss))
            if self.swa:
                if epoch > self.swa_start:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
            else:
                self.scheduler.step(np.mean(self.writer.val_loss))
            self.writer.add_scalar('Validation Loss', np.mean(self.writer.val_loss), epoch)

    def inference(self, model_name='best_model.pt', plot_output=False, save_preds=False):
        #~ Model Inference
        print('Inference...')
        if self.model_path is None:
            self.model.load_state_dict(torch.load(self.output_path + model_name))
        else:
            self.model.load_state_dict(
                torch.load(self.model_path + model_name))
        self.model.eval()
        all_ids = []
        all_masks = []
        with torch.set_grad_enabled(False):
            for idx, data in enumerate(tqdm(self.test_dataLoader)):
                #* Load data
                img = data['sag_image'].to(
                    self.device, dtype=torch.float32)
                ids = data['id']
                #* Get predictions
                pred_seg = self.model(img)
               
                if plot_output:
                    #* Plot predictions
                    self.plot_mask(ids, pred_seg, img)
                all_ids.append(ids)
                all_masks.append(pred_seg.cpu().numpy())

        all_ids = np.concatenate(all_ids, axis=0)
        all_masks= np.concatenate(all_masks, axis=0)

        print(all_ids.shape, all_masks.shape)
        if save_preds:
            #** Save predictions to npz file for post-processing
            print('SAVING PREDICTIONS...')
            np.savez(self.output_path + f'{model_name.split(".")[0]}_preds.npz', ids=all_ids,
                        masks=all_masks)
        else:
            return all_ids, all_masks

    def viz_model(self, output_path='./logs/'):
        #~ View model architecture
        print('Vis model')
        input_shape = (4, 3, 512, 256) 
        model = self.model
        #* Create placeholder
        x = Variable(torch.randn(input_shape)).to('cuda')
        mask, classes = model(x, x)
        #* Only follow path of coordinates
        graph = make_dot(mask.mean(), params=dict(model.named_parameters()))
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
    
    def plot_mask(self, names, pred, img):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.axis('off')
        for idx in np.arange(len(names)):
            arr = pred[idx].cpu().numpy()
            arr = self.sigmoid(arr)
            img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            img = self.norm_img(img)
            ax.imshow(img)
            ax.imshow(arr[idx], alpha=0.5)
            fig.savefig(self.output_path + f'masks/{names[idx]}.png')
            plt.clf()
        plt.close()


