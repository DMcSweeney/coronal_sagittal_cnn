"""
Locator (labelling) model training loop
"""

import os
from einops.einops import rearrange
import numpy as np
import matplotlib
from scipy.ndimage.morphology import distance_transform_bf
from torch.functional import norm
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
from utils.customLosses import EarlyStopping, FocalLoss, dice_loss, kl_reg, multi_class_dice, edgeLoss

class Labeller():
    """
    ~Class for training vertebrae labelling
    @params: 
      dir_name = directory name used for splitting tensorboard runs.   
    """
    def __init__(self, training=None, validation=None, testing=None, dir_name=None, device="cuda:0", 
                    batch_size=4, n_outputs=13, learning_rate=3e-3, num_epochs=200, output_path='./outputs/', 
                    model_path=None, SWA=False, classifier=False, norm_coords=False, early_stopping=False):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.train_dataLoader = training
        self.val_dataLoader = validation
        self.test_dataLoader = testing
        self.classifier = classifier
        self.norm_coords = norm_coords
        self.model = cm2.labelNet(
            n_outputs=n_outputs, classifier=classifier, norm_coords=norm_coords).cuda()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.es = EarlyStopping(patience=75) if early_stopping else None

        #* Losses 
        self.ce = edgeLoss(self.device).cuda()
        self.ce_weight = 50
        self.mse = dsntnn.euclidean_losses
        self.mse_weight = 75
        self.kl = kl_reg
        self.kl_weight = 1
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2])).to(device) if classifier else None
        self.bce_weight = 5
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True, patience=30)

        self.writer = cw2.customWriter(
            log_dir=f'./runs/{dir_name}', batch_size=batch_size, num_classes=n_outputs)
        self.num_epochs = num_epochs
        
        self.best_loss = 10000 # Initialise loss for saving best model
        self.output_path = output_path
        self.model_path = model_path
        
        #** Stochastic Weight Averaging (https://pytorch.org/docs/stable/optim.html#putting-it-all-together)
        self.swa = SWA
        self.swa_model = swa.AveragedModel(self.model) if SWA else None
        self.swa_scheduler = swa.SWALR(
            self.optimizer, swa_lr=0.05) if SWA else None  # LR set to large value
        self.swa_start = 100 if SWA else None  # START EPOCH from SWA

        self.ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                              'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']

    @staticmethod
    def masked_mean(x, labels):
        a = x[:, 1:]*labels  # * Ignore background
        x = x.sum(-1) / (a > 0).float().sum(-1)  # * Masked average
        return x.mean() # * Batchmean

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

    def forward(self, num_epochs=200, model_name='best_model.pt'):
        self.writer.init_losses(keys=['train_loss', 'val_loss', 'ce', 'kl', 'bce', 'mse'])

        #~ Forward pass def
        for epoch in range(num_epochs+1):
            self.writer.epoch = epoch
            # self.kl_weight = epoch/num_epochs
            # self.ce_weight = 1-(epoch/num_epochs)

            print(f'Epoch: {epoch}/{num_epochs}')
            #~TRAINING
            self.train(epoch=epoch, write2tensorboard=True, writer_interval=20)
            #~ VALIDATION
            self.validation(epoch=epoch, write2tensorboard=True, writer_interval=5, write_gif=False)
            #* Save best model + check early stopping
            stop = self.save_best_model(model_name=model_name)
            if stop:
                break
        
        #* Update batch norm stats. for SWA model
        if self.swa:
            print('Updating batch norm stats')
            for data in tqdm(self.train_dataLoader):
                img = data['image'].to(self.device, dtype=torch.float32)
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
            img = data['image'].to(self.device, dtype=torch.float32)
            mask = data['mask'].to(self.device, dtype=torch.long)
            heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
            keypoints, labels = data['keypoints'].to(self.device, dtype=torch.float32), \
                data['labels'].to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad() #* Reset gradients

            #* Sharpen heatmap
            heatmap = self.writer.sharpen_heatmap(heatmap)
            if self.classifier:
                pred_seg, pred_heatmap, pred_coords, pred_labels = self.model(img)
                bce = self.bce(pred_labels, labels)
            else:
                pred_seg, pred_heatmap, pred_coords = self.model(img)

            #* Losses
            ce = self.ce(pred_coords, keypoints, labels)
            kl = self.kl(pred_heatmap, heatmap[: , 1: ])
            kl = dsntnn.average_loss(kl, mask=labels)
            mse = self.mse(pred_coords, keypoints)
            mse = dsntnn.average_loss(mse, mask=labels)
            loss = ce*self.ce_weight + kl*self.kl_weight + mse*self.mse_weight
            if self.classifier:
                loss += bce
            self.writer.losses['train_loss'].append(loss.item())
            #* Optimiser step
            loss.backward()
            self.optimizer.step()

            if write2tensorboard:
                # ** Write inputs to tensorboard
                if epoch % writer_interval ==0  and idx == 0:
                    self.writer.plot_mask('Ground-Truth', img=img, prediction=mask)
                    self.writer.plot_heatmap('Ground-Truth heatmap', img=img, heatmap=heatmap, 
                    apply_softmax=False, norm_coords=True, labels=labels)
            
        print('Train Loss:', np.mean(self.writer.losses['train_loss']))
        self.writer.add_scalar('Training Loss', np.mean(
            self.writer.losses['train_loss']), epoch)
            
    def validation(self, epoch, write2tensorboard=True, writer_interval=10, write_gif=False):
        #~Validation loop
        with torch.set_grad_enabled(False):
            print('Validation...')
            for idx, data in enumerate(tqdm(self.val_dataLoader)):
                #* Load data
                img = data['image'].to(self.device, dtype=torch.float32)
                mask = data['mask'].to(self.device, dtype=torch.long)
                heatmap = data['heatmap'].to(self.device, dtype=torch.float32)
                keypoints, labels = data['keypoints'].to(
                    self.device, dtype=torch.float32), data['labels'].to(self.device, dtype=torch.float32)

                #* Sharpen heatmap
                heatmap = self.writer.sharpen_heatmap(heatmap)
                if self.classifier:
                    pred_seg, pred_heatmap, pred_coords, pred_labels = self.model(
                        img, writer=self.writer)
                    bce = self.bce(pred_labels, labels)
                else:
                    pred_seg, pred_heatmap, pred_coords = self.model(img)
                #* Losses
                ce = self.ce(pred_coords, keypoints, labels)
                kl = self.kl(pred_heatmap, heatmap[: , 1: ])
                kl = dsntnn.average_loss(kl, mask=labels)
                mse = self.mse(pred_coords, keypoints)
                mse = dsntnn.average_loss(mse, mask=labels)

                loss = ce*self.ce_weight + kl*self.kl_weight + mse*self.mse_weight
                if self.classifier:
                    loss += bce * self.bce_weight
                    self.writer.losses['bce'].append(bce.item())

                self.writer.losses['val_loss'].append(loss.item())
                self.writer.losses['kl'].append(kl.item())
                self.writer.losses['ce'].append(ce.item())
                self.writer.losses['mse'].append(mse.item())

                if write2tensorboard:
                    #* Write predictions to tensorboard
                    if epoch % writer_interval == 0 and idx==0:
                        if self.norm_coords:
                            pred_coords = dsntnn.normalized_to_pixel_coordinates(pred_coords, size=(512, 512))
                            keypoints = dsntnn.normalized_to_pixel_coordinates(keypoints, size=(512, 512))

                        self.writer.plot_prediction(f'Heatmap Predictions', img=img, prediction=pred_heatmap, 
                                                    ground_truth=heatmap, type_='heatmap', 
                                                    apply_norm=False, coords=pred_coords, 
                                                    gt_coords=keypoints, labels=pred_labels)

                        self.writer.plot_prediction(f'Mask Predictions', img=img, prediction=pred_seg,
                                                    ground_truth=mask, type_='mask', apply_norm=False, 
                                                    )
                        

            print('Validation Loss:', np.mean(self.writer.losses['val_loss']))
            print(
                f'\n CE:{np.mean(self.writer.losses["ce"])*self.ce_weight}-KL:{np.mean(self.writer.losses["kl"])*self.kl_weight}-MSE:{np.mean(self.writer.losses["mse"])*self.mse_weight}-BCE:{np.mean(self.writer.losses["bce"])*self.bce_weight}')



            if self.swa:
                if epoch > self.swa_start:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
            else:
                self.scheduler.step(np.mean(self.writer.losses['val_loss']))
            #* Write losses
            self.writer.add_scalar('Validation Loss', np.mean(self.writer.losses['val_loss']), epoch)
            self.writer.add_scalar('KL Div', np.mean(self.writer.losses['kl']), epoch)
            self.writer.add_scalar('CE loss', np.mean(self.writer.losses['ce']), epoch)
            self.writer.add_scalar('MSE loss', np.mean(self.writer.losses['mse']), epoch)
            if self.classifier:
                self.writer.add_scalar('BCE loss', np.mean(self.writer.losses['bce']), epoch)

    def inference(self, model_name='best_model.pt', plot_output=False, save_preds=False):
        #~ Model Inference
        print('Inference...')
        if self.model_path is None:
            self.model.load_state_dict(torch.load(self.output_path + model_name))
        else:
            self.model.load_state_dict(torch.load(self.model_path + model_name))
        self.model.eval()
        all_ids = []
        all_masks = []
        all_heatmaps = []
        all_labels = []
        all_coords = []
        with torch.set_grad_enabled(False):
            for idx, data in enumerate(tqdm(self.test_dataLoader)):
                #* Load data
                img = data['image'].to(
                    self.device, dtype=torch.float32)
                ids = data['id']
                keypoints = data['keypoints'].to(self.device, dtype=torch.float32)
                #* Get predictions
                if self.classifier:
                    pred_seg, pred_heatmap, pred_coords, pred_labels = self.model(img)
                else:
                    pred_seg, pred_heatmap, pred_coords = self.model(img)

                pred_coords= dsntnn.normalized_to_pixel_coordinates(pred_coords, size=(512, 512))

                if plot_output:
                    keypoints = dsntnn.normalized_to_pixel_coordinates(keypoints, size=(512, 512))
                    os.makedirs(os.path.join(self.output_path, 'sanity'), exist_ok=True)
                    #* Plot predictions
                    self.plot_predictions(ids, img, coords=pred_coords, gt_coords=keypoints)


                all_ids.append(ids)
                all_masks.append(pred_seg.cpu().numpy())
                all_heatmaps.append(pred_heatmap.cpu().numpy())
                all_coords.append(pred_coords.cpu().numpy())
                if self.classifier:
                    all_labels.append(pred_labels.cpu().numpy())

        all_ids = np.concatenate(all_ids, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        all_heatmaps = np.concatenate(all_heatmaps, axis=0)
        all_coords = np.concatenate(all_coords, axis=0)
        if self.classifier:
            all_labels = np.concatenate(all_labels, axis=0)
            print(all_ids.shape, all_masks.shape, all_heatmaps.shape, all_coords.shape, all_labels.shape)
        else:
            print(all_ids.shape, all_masks.shape, all_heatmaps.shape, all_coords.shape)
        
        if save_preds:
            #** Save predictions to npz file for post-processing
            print('SAVING PREDICTIONS...')
            if self.classifier:
                np.savez(self.output_path + f'{model_name.split(".")[0]}_preds.npz', ids=all_ids,
                            coords=all_coords, heatmaps=all_heatmaps, masks=all_masks, labels=all_labels)
            else:
                np.savez(self.output_path + f'{model_name.split(".")[0]}_preds.npz', ids=all_ids,
                         coords=all_coords, heatmaps=all_heatmaps, masks=all_masks)
        else:
            if self.classifier:
                return all_ids, all_masks, all_heatmaps, all_coords, all_labels
            else:
                return all_ids, all_masks, all_heatmaps, all_coords

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

    def save_best_model(self, model_name='best_model.pt'):
        #~ Check if latest validation is min. If True, save.
        loss1 = np.mean(self.writer.losses['val_loss'])
        is_best = loss1 < self.best_loss
        self.best_loss = min(loss1, self.best_loss)
        if is_best:
            print('Saving best model')
            torch.save(self.model.state_dict(),
            self.output_path + model_name)
        
        if self.es is None:
            return False
        else:
            if self.es.step(torch.tensor([loss1])):
                return True
            else:
                return False
        
    def plot_predictions(self, names, img, coords=None, gt_coords=None):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].axis('off')
        
        ax[1].axis('off')
        for idx in np.arange(len(names)):
            img = rearrange(img[idx].cpu(), 'c h w -> h w c')
            img = img.numpy().squeeze()
            img = self.norm_img(img)
            ax[0].set_title('Ground-Truth')
            ax[0].imshow(img)
            ax[1].set_title('Prediction')
            ax[1].imshow(img)

            if coords is not None:
                for i, vert in enumerate(self.ordered_verts):
                    y, x = coords[idx, i].cpu().numpy()
                    ax[1].scatter(x, y, marker='+', c='r', s=15)
                    ax[1].text(x, y, vert, c='r', size=15)
            if gt_coords is not None:
                for i, vert in enumerate(self.ordered_verts):
                    y, x = gt_coords[idx, i].cpu().numpy()
                    ax[0].scatter(x, y, marker='+', c='r', s=15)
                    ax[0].text(x, y, vert, c='r', size=15)
            fig.savefig(self.output_path + f'sanity/{names[idx]}.png')
            plt.clf()
        plt.close()
