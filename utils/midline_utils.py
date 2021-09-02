"""
Set of useful functions for converting between midline predictions to extracting a sagittal midline
"""
from projections import pad_image
import numpy as np
import os
from scipy import interpolate
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import projections as proj

def sigmoid(x):
    return 1/(1+np.exp(-x))

def mask2midline(pred, N=20):
    #~ Convert mask to midline 
    #@params:
    #* N: Number of points to sample
    #* Get center of contours for line fitting
    #!pred = sigmoid(pred)
    x_template = np.zeros_like(pred)
    y_template = np.zeros_like(pred)
    for x in range(x_template.shape[0]):
        x_template[:, x] = x
        y_template[x] = x
    norm_pred = np.where(pred < 0.5, 0, 1.0)  # Binarise mask
    exp_x = x_template*norm_pred
    x_list = []
    y_list = []
    #* Calculate expectation
    for i in range(pred.shape[0]):
        y_preds = np.mean(exp_x[i]).astype(int)
        if y_preds != 0:
            x_list.append(np.mean(exp_x[i][exp_x[i] != 0]))
            y_list.append(i)
    #* Sample list at even intervals
    samp_y = []
    samp_x = []
    for n in range(N):
        int_ = len(y_list) // N
        samp_y.append(y_list[n*int_])
        samp_x.append(x_list[n*int_])
    #* Fit spline
    tck = interpolate.splrep(samp_y, samp_x, k=1, s=0)
    xd = np.linspace(min(samp_y), max(samp_y), int(max(samp_y)-min(samp_y)))
    fit = interpolate.splev(xd, tck, der=0)
    #* Extend top and bottom
    top = [fit[0]]*int(min(samp_y))
    bot = [fit[-1]]*(512-int(max(samp_y)))
    top.extend(fit)
    top.extend(bot)
    if len(top) != 512:
        top.append(fit[-1])
    return top


def pad_image(img, output_shape=(626, 452)):
    """
    Insert into array of fixed size - such that all inputs to model have same dimensions
    """
    padding = [(s-x)//2.0 for x, s in zip(output_shape, img.GetSize())]
    output_origin = img.TransformContinuousIndexToPhysicalPoint(padding)
    pad = sitk.ResampleImageFilter()
    pad.SetOutputSpacing(img.GetSpacing())
    pad.SetSize(output_shape)
    pad.SetDefaultPixelValue(-1024)
    pad.SetOutputOrigin(output_origin)
    pad.SetOutputDirection(img.GetDirection())
    return pad.Execute(img), padding

def plot_midline(name, fit, slice_):
    img_path = f'/data/PAB_data/images_coronal/mip/{name}.png'
    img = np.array(Image.open(img_path))
    xd = np.arange(0, 512, step=1)  
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(img, cmap='gray')
    ax[0].plot(fit, xd, c='y', lw=2.5)
    ax[1].imshow(slice_, cmap='gray')
    fig.subplots_adjust(wspace=0.01)
    fig.savefig(f'./midline_output/sagittal_midline/{name}.png', transparent=True)

def get_ROI(vol, fit, width=4):
    #* Extract thick sagittal slice
    roi = np.zeros((512, 2*width, 512))
    roi -= 1024
    for i, y in enumerate(fit):
        sample = vol[int(y)-width:int(y)+width, i]
        roi[i] = sample
    return roi

def get_sagittal_slice(name, fold, fit):
    vol_path = f'/data/PAB_data/volume_folds/q{fold}/{name}.nii'
    vol = sitk.ReadImage(vol_path)
    #* Resample volume
    vol, _ = proj.resample(vol, (1.25, 1.25, 1.25))
    #* Pad to 512, 512, 512
    vol,  _ = pad_image(vol, (512, 512, 512))
    vol = sitk.GetArrayFromImage(vol)
    #* Extract thick slice
    thick_slice = get_ROI(vol, fit)
    norm_slice = proj.WL_norm(sitk.GetImageFromArray(thick_slice), window=600, level=200)
    norm_slice = proj.normalize(norm_slice)
    slice_ = np.max(sitk.GetArrayFromImage(norm_slice), axis=1)
    return slice_

def extract_sagittal_projections(ids, masks, fold, output_path, save_slice=False, plot_slice=False):
    #~ Main function for conversion
    for id_, mask in zip(ids, masks):
        print(id_)
        #* Convert to midline fit
        fit = mask2midline(mask[0], N=15)
        #* Convert back to original frame   
        slice_ = get_sagittal_slice(id_, fold, fit)
        if save_slice:
            np.save(os.path.join(output_path, f'{id_}.npy'), slice_)
        if plot_slice:
            #* Plot predictions
            plot_midline(id_, fit, slice_)
