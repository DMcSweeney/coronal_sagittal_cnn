"""
Script containing custom model
W/ DSNTNN layer
"""
import torch
import torch.nn as nn
import torchvision.models as models
import dsntnn
import random
import customDSNT as cstmDSNT
random.seed(60)

class customResnet(nn.Module):
    def __init__(self, n_outputs):
        """
        seg_outputs = outputs for segmentation map  (should be total verts + 1 - for background)
        class_outputs = outputs for classifier (should be total verts)
        """

        super(customResnet, self).__init__()
        # Model
        # Control std of heatmaps used for calculating JS divergence
        #self.sigma = nn.Parameter(torch.tensor([random.uniform(1, 10) for n in range(n_outputs)]), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([random.uniform(1, 10)]), requires_grad=True)
        self.model = self.prep_model(1)
        self.hm_conv = nn.Conv2d(4, n_outputs, kernel_size=1, bias=False)

    def prep_model(self, num_outputs):
        """
        Load pre-trained weights
        """

        pt_model = models.segmentation.fcn_resnet101(pretrained=True)
        model = models.segmentation.fcn_resnet101(
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


    def forward(self, x):
        seg_out = self.model(x)
        norm_seg_out = seg_out['out']#torch.nn.functional.softmax(seg_out['out'], dim=1)
        heatmap_in = torch.cat([x, norm_seg_out], dim=1)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(heatmap_in)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        #heatmaps = cstmDSNT.sharpen_heatmaps(heatmaps, alpha=10)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)
        return coords, heatmaps, unnormalized_heatmaps, seg_out['out']
