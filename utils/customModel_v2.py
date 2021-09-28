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
from types import MethodType
from torch.quantization import QuantStub, DeQuantStub
from torchvision.models.resnet import BasicBlock
from segmentation_models_pytorch.unet.decoder import DecoderBlock
from utils.modules import ResidualBlock
from einops import rearrange

random.seed(60)

#** --------------------------------------LABEL NET -------------------------
class labelNet(nn.Module):
    def __init__(self, n_outputs, classifier=False, norm_coords=False):
        super(labelNet, self).__init__()
        self.input_size = (512, 512)
        self.decoder_channels = [256, 128, 64, 32, 16]
        self.decoder_out_channels = [4*x for x in self.decoder_channels]
        # For classifier at end of segmentation head
        self.aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation=None,      # activation function, default is None
            classes=n_outputs-1,                 # define number of output labels
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

        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
        self.heatmap_head = ResidualBlock(in_channels=16, out_channels=14, kernel_size=1, 
                    downsample=nn.Conv2d(in_channels=16, out_channels=14, kernel_size=1, stride=1))
        self.classifier = self.model.classification_head if classifier else None
        self.norm_coords = norm_coords
        self.heatmap2onedim = nn.Conv2d(
            14, n_outputs, kernel_size=(1, self.input_size[1]), bias=False)

        self.basic_block = ResidualBlock(in_channels=13, out_channels=13, kernel_size=1,
                                         downsample=nn.Conv2d(in_channels=13, out_channels=13, kernel_size=1, stride=1))

    @staticmethod
    def sharpen_heatmap(heatmap, alpha=2):
        sharp_map = heatmap ** alpha
        eps = 1e-24
        flat_map = sharp_map.flatten(2).sum(-1)[..., None, None]
        flat_map += eps
        return sharp_map/flat_map

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def forward(self, x, writer=None):
        Z = self.encoder.forward(x) #* Z=latent variable
        Y = self.decoder.forward(*Z) #* Y=prediction 

        if self.classifier is not None:
            # * Classifier at end of encoder
            class_out = self.classifier(Z[-1])

        segmentation = self.segmentation_head.forward(Y)
        heatmap = self.heatmap_head.forward(Y) 
        heatmap = F.relu(heatmap) 
        #* Element-wise prod. of mask + heatmap
        map_ = F.softmax(segmentation, dim=1)*heatmap #* Softmax along channel axis
        if writer is not None:
            ...
            # writer.plot_prediction(f'H x S', img=x, prediction=map_, type_='heatmap', ground_truth=None,apply_norm=True)
            # writer.plot_prediction(f'Sigmoid mask', img=x, prediction=F.softmax(segmentation, dim=1), 
            #                             type_='heatmap', ground_truth=None, apply_norm=False)

        # #* Reshape arrays
        map_ = rearrange(map_[:, 1:], 'b c h w -> b c (h w)') #* Ignore background
        #* Filter out channels based on classifier
        class_map = self.sigmoid(class_out[..., None])*map_
        class_map = rearrange(class_map, 'b c (h w) -> b c h w', h=512, w=512)

        norm_map = self.basic_block.forward(class_map)
        norm_map = F.relu(norm_map)
        # #norm_map = self.sigmoid(norm_map)
        # print(map_[:, 1:].shape, norm_map.shape)
        #norm_map = dsntnn.flat_softmax(map_[:, 1:].contiguous())
        norm_map = self.sharpen_heatmap(norm_map, alpha=2)
        
        if writer is not None:
            ...
        #     writer.plot_prediction('Coordinate map', img=x, prediction=norm_map, type_='heatmap', 
        #     ground_truth=None, apply_norm=False)

        coords = dsntnn.dsnt(norm_map, normalized_coordinates=self.norm_coords)
        if self.classifier:
            return segmentation, norm_map, coords, class_out

        return segmentation, norm_map, coords

#** --------------------SIAMESE UNET --------------------
class customSiameseUNet(nn.Module):
    def __init__(self, n_outputs, input_size=(512,512)):
        """
        seg_outputs = outputs for segmentation map  (should be total verts + 1 - for background)
        class_outputs = outputs for classifier (should be total verts)
        """
        super(customSiameseUNet, self).__init__()
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


##** --------------------------------------------------CUSTOM UNET --------------------------------------


class customUNet(nn.Module):
    def __init__(self, n_outputs, classifier=False, input_size=(512,512)):
        """
        seg_outputs = outputs for segmentation map  (should be total verts + 1 - for background)
        class_outputs = outputs for classifier (should be total verts)
        """
        super(customUNet, self).__init__()
        self.input_size = input_size
        self.decoder_channels = [256, 128, 64, 32, 16]
        # For classifier at end of segmentation head
        self.aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation=None,      # activation function, default is None
            classes=n_outputs-1,                 # define number of output labels
        )
        # Model
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            encoder_depth=5,
            in_channels=3,
            #classes=n_outputs,
            classes=1,
            aux_params=self.aux_params,
            decoder_channels=tuple(self.decoder_channels)
        )
        #* Decompose model
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
        self.classifier = self.model.classification_head if classifier else None


    def forward(self, x):
        #* Output are features at every spatial resolution
        out = self.encoder.forward(x)
        #* Upscale to match input spatial resolution
        decoder_output = self.decoder.forward(*out)

        #* Get correct number of channels
        seg_out = self.segmentation_head.forward(decoder_output)

        if self.classifier is not None:
            class_out = self.classifier(out[-1])
            return seg_out, class_out

        return seg_out

#* ------------------- Quantization-aware  -----------


class qat_UNet(nn.Module):
    #~ UNet ready for quantization-aware training
    def __init__(self, n_outputs, classifier=False, input_size=(512, 512)):
        """
        seg_outputs = outputs for segmentation map  (should be total verts + 1 - for background)
        class_outputs = outputs for classifier (should be total verts)
        """
        super(qat_UNet, self).__init__()
        self.input_size = input_size
        self.decoder_channels = [256, 128, 64, 32, 16]
        # For classifier at end of segmentation head
        self.aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation=None,      # activation function, default is None
            classes=n_outputs,                 # define number of output labels
        )
        # Model
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            encoder_depth=5,
            in_channels=3,
            classes=1,
            aux_params=self.aux_params,
            decoder_channels=tuple(self.decoder_channels)
        )
        #* Decompose model
        self.quant = QuantStub()
        self.unquant = DeQuantStub()
        #* Quantize encoder
        self.encoder = self.model.encoder
        self.encoder.quant = QuantStub()  # * Converts float tensors to int
        self.encoder.unquant = DeQuantStub() #* Converts back to float
        self.encoder.forward = MethodType(self.encoder_forward, self.encoder)

        #* Quantize decoder
        self.decoder = self.model.decoder
        self.decoder.quant = QuantStub()
        self.decoder.unquant = DeQuantStub()
        self.decoder.forward = MethodType(self.decoder_forward, self.decoder)

        #* Seg. head + classifier are sequential not modules 
        self.segmentation_head = self.model.segmentation_head
        self.classifier = self.model.classification_head if classifier else None

    def forward(self, x):
        #* Output are features at every spatial resolution
        out = self.encoder.forward(x)
        #* Upscale to match input spatial resolution
        decoder_output = self.decoder.forward(*out)

        #* Get correct number of channels
        decoder_output = self.quant(decoder_output)
        seg_out = self.segmentation_head.forward(decoder_output)
        seg_out = self.unquant(seg_out)
        if self.classifier is not None:
            class_in = self.quant(out[-1])
            class_out = self.classifier(class_in)
            class_out = self.unquant(class_out)
            return seg_out, class_out
        return seg_out

    def fuse_model(self):
        #* Prepare model for quantization-aware training
        #* Only fuse encoder basic blocks
        #TODO Figure out how to fuse remaining layers
        for m in self.encoder.modules():
            if type(m) == BasicBlock:
                torch.quantization.fuse_modules(m, ['conv1', 'bn1', 'relu'], inplace=True)


    @staticmethod
    def encoder_forward(self, x):
        #~ Quantized impl. of encoder forward method
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = self.quant(x) #* Quantize
            x = stages[i](x)
            x = self.unquant(x) #* Back to float
            features.append(x)

        return features

    @staticmethod
    def decoder_forward(self, *features):
        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]
        x = self.quant(head)
        x = self.center(x)
        x = self.unquant(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = self.quant(x)
            x = decoder_block(x, skip)
            x = self.unquant(x)
        return x
