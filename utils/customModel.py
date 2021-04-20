"""
Script containing custom model
W/ DSNTNN layer
"""
import torch
import torch.nn as nn
import torchvision.models as models
import dsntnn
import torch.nn.functional as F
import random
import segmentation_models_pytorch as smp

random.seed(60)

class customResnet(nn.Module):
    def __init__(self, n_outputs, input_size):
        """
        seg_outputs = outputs for segmentation map  (should be total verts + 1 - for background)
        class_outputs = outputs for classifier (should be total verts)
        """
        self.input_size = input_size
        super(customResnet, self).__init__()
        # For classifier at end of segmentation head
        self.aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=n_outputs,                 # define number of output labels
        )
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_outputs,
            aux_params=self.aux_params
        )

        print(self.model)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.head = self.model.segmentation_head
        self.hm_conv = nn.Conv2d(13, n_outputs, kernel_size=(1, self.input_size[1]), bias=False)

    def prep_model(self, num_outputs):
        """
        Load pre-trained weights
        """
        pt_model = models.segmentation.fcn_resnet50(pretrained=False)
        model = models.segmentation.fcn_resnet50(
            pretrained=False, num_classes=num_outputs)

        pt_dict = pt_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {}
        for key, val in pt_dict.items():
            if key in model_dict:
                if val.shape == model_dict[key].shape:
                    pretrained_dict[key] = val
                else:
                    print("Shapes don't match")
                    continue
            else:
                print("key not in dict")
                continue
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # Load new state dict
        model.load_state_dict(model_dict)
        return model


    def forward(self, sag_x, cor_x):
        sag_out = self.encoder(sag_x)
        cor_out = self.encoder(cor_x)
        print(len(sag_out))
        print(sag_out[0].shape)
        seg_out = self.head(self.decoder(torch.mul(sag_out['out'], cor_out['out'])))
        print(seg_out.shape)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnorm_heatmap = self.hm_conv(seg_out) # 1-Dim Heatmap (BxCxHx1)
        print(unnorm_heatmap.shape)
        # 3. Softmax across HxW
        heatmap = dsntnn.flat_softmax(unnorm_heatmap)
        # Coordinate from heatmap (Return )
        coords= dsntnn.dsnt(heatmap, normalized_coordinates=False)
        
        #Global Average pooling
        pred = F.avg_pool2d(seg_out, kernel_size=seg_out.size()[2:])

        return unnorm_heatmap, seg_out, coords[..., 1], pred
