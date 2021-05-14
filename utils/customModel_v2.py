"""
Script containing custom model
W/ DSNTNN layer
"""
from torchsummary import summary
from torch.autograd import Variable
import graphviz
from torchviz import make_dot
import torch
import torch.nn as nn
import torchvision.models as models
import dsntnn
import torch.nn.functional as F
import random
import segmentation_models_pytorch as smp

random.seed(60)


class customUNet(nn.Module):
    def __init__(self, n_outputs, input_size=(512,512)):
        """
        seg_outputs = outputs for segmentation map  (should be total verts + 1 - for background)
        class_outputs = outputs for classifier (should be total verts)
        """
        super(customUNet, self).__init__()
        self.input_size = input_size
        self.decoder_channels = [256, 128, 64, 32, 16]
        self.decoder_out_channels = [4*x for x in self.decoder_channels]
        # For classifier at end of segmentation head
        self.aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=n_outputs,                 # define number of output labels
        )
        # Model
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            encoder_depth=5,
            in_channels=3,
            classes=n_outputs,
            aux_params=self.aux_params,
            decoder_channels=tuple(self.decoder_channels)
        )
        #!! Encoder- CHECK THIS SETUP
        self.encoder = self.model.encoder
        # self.sag_encoder = self.model.encoder
        # self.cor_encoder = self.model.encoder
        # Update decoder 
        self.decoder = smp.unet.decoder.UnetDecoder(
            encoder_channels=tuple([2*x for x in self.encoder.out_channels]),
            decoder_channels=tuple(self.decoder_out_channels),
            n_blocks=5,
            use_batchnorm=True,
            center=False, #!! True if using vgg
            attention_type=None,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=self.decoder_out_channels[-1],
            out_channels=n_outputs
            )
        self.classifier = self.model.classification_head
        self.hm_conv = nn.Conv2d(13, n_outputs, kernel_size=(1, self.input_size[1]), bias=False)

    def forward(self, sag_x, cor_x):
        #* Output are features at every spatial resolution
        sag_out = self.encoder.forward(sag_x)
        cor_out = self.encoder.forward(cor_x)
    
        if self.classifier is not None:
            sag_class = self.classifier(sag_out[-1])
            cor_class = self.classifier(cor_out[-1])
            # print(sag_class.shape, cor_class.shape)
            mean_class = (sag_class+cor_class)/2

        #* Combine Sagittal + coronal at each resolution
        output = [torch.cat([sag,cor], dim=1) for sag, cor in zip(sag_out, cor_out)]

        #output = [sag*cor for sag, cor in zip(sag_out, cor_out)]
        #* Upscale to match input spatial resolution
        decoder_output = self.decoder.forward(*output)

        #* Get correct number of channels
        seg_out = self.segmentation_head.forward(decoder_output)

        #* Convert to 1D heatmap
        unnorm_heatmap = self.hm_conv(seg_out) 
        heatmap = dsntnn.flat_softmax(unnorm_heatmap[..., 0])
        coords = dsntnn.dsnt(heatmap, normalized_coordinates=True)
        return seg_out, heatmap, coords[..., 0], mean_class

    @staticmethod
    def sharpen_heatmaps(heatmaps, alpha):
        #* https://github.com/anibali/dsntnn/blob/master/dsntnn/__init__.py
        sharpened_heatmaps = heatmaps ** alpha
        sharpened_heatmaps /= sharpened_heatmaps.flatten(2).sum(-1, keepdim=True)
        return torch.unsqueeze(sharpened_heatmaps, dim=-1)
