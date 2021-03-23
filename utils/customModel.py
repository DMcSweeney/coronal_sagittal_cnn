"""
Script containing custom model
W/ DSNTNN layer
"""
import torch
import torch.nn as nn
import torchvision.models as models
import dsntnn
import random
random.seed(60)

class customResnet(nn.Module):
    def __init__(self, n_outputs, input_size):
        """
        seg_outputs = outputs for segmentation map  (should be total verts + 1 - for background)
        class_outputs = outputs for classifier (should be total verts)
        """
        self.input_size = input_size
        super(customResnet, self).__init__()
        # Model
        # Control std of heatmaps used for calculating JS divergence
        #self.sigma = nn.Parameter(torch.tensor([random.uniform(1, 10)]), requires_grad=True)
        self.model = self.prep_model(n_outputs)
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
        sag_out = self.model(sag_x)
        cor_out = self.model(cor_x)
        seg_out = (sag_out['out'] + cor_out['out'])/2
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        # 3. Normalize the heatmaps
        heatmap = self.hm_conv(seg_out)
        return heatmap, seg_out
