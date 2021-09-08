"""
Script for converting 3D volume to different projections (MIP, avg, rms)
"""

import matplotlib
matplotlib.use('Agg')
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv
import pandas as pd
from einops import rearrange
from itertools import groupby
from collections import Counter

ordered_verts = ['T4', 'T5', 'T6', 'T7', 'T8', 'T9',
                 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4']


def resample(image, new_spacing):
    """
    Resample image to new resolution with pixel dims defined by new_spacing
    """
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = image.GetSpacing()
    ratio = [x/y for x, y in zip(orig_spacing, new_spacing)]
    new_size = orig_size*ratio
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    return resample.Execute(image), ratio

def WL_norm(img, window=1000, level=700):
    """
    Apply window and level to image
    """

    maxval = level + window/2
    minval = level - window/2
    wl = sitk.IntensityWindowingImageFilter()
    wl.SetWindowMaximum(maxval)
    wl.SetWindowMinimum(minval)
    out = wl.Execute(img)
    return out

def maximum_projection(img, dim=0):
    """
    Maximum intensity projection along dim
    """
    mip = sitk.MaximumProjectionImageFilter()
    mip.SetProjectionDimension(dim)
    return mip.Execute(img)

def average_projection(img, dim=0):
    """
    Average projection along dim
    """
    avg = sitk.MeanProjectionImageFilter()
    avg.SetProjectionDimension(dim)
    return avg.Execute(img)

def standard_deviation(img, dim=0):
    """
    Standard deviation projection - similar to RMS projection?
    """
    std = sitk.StandardDeviationProjectionImageFilter()
    std.SetProjectionDimension(dim)
    return std.Execute(img)

def normalize(img, min=0, max=255):
    """
    Cast float image to int, and normalise to [0, 255]
    """
    img = sitk.Cast(img, sitk.sitkInt32)
    norm = sitk.RescaleIntensityImageFilter()
    norm.SetOutputMaximum(max)
    norm.SetOutputMinimum(min)
    return norm.Execute(img)

def pad_image(img, output_shape=(626, 452)):
    """
    Insert into array of fixed size - such that all inputs to model have same dimensions
    """
    padding = [(s-x)//2.0 for x, s in zip(output_shape, img.GetSize())]
    output_origin = img.TransformContinuousIndexToPhysicalPoint(padding)
    pad = sitk.ResampleImageFilter()
    pad.SetOutputSpacing(img.GetSpacing())
    pad.SetSize(output_shape)
    pad.SetOutputOrigin(output_origin)
    pad.SetOutputDirection(img.GetDirection())
    return pad.Execute(img), padding

def post_projection(img, new_spacing, output_shape=(626, 452)):
    """
    Post-processing for projections, resample to isotropic grid and make standardised shape
    """
    resampled_img, scale = resample(img, new_spacing)
    padded_img, padding = pad_image(resampled_img, output_shape)
    return padded_img, scale, padding

def plot_projection(img, name, path):
    #! Transpose needed to plot them in correct orient.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')
    arr = sitk.GetArrayFromImage(img)
    if 'coronal' in path:
        im = Image.fromarray(arr.astype(np.uint8).T)
    elif 'sagittal' in path:
        im = Image.fromarray(arr.astype(np.uint8))
    else:
        raise ValueError
    os.makedirs(path, exist_ok=True)
    im.save(f'{path}{name}.png')
    plt.close()

def write_threeChannel(images, name, path, output_shape=(626, 452)):
    """
    Write projections to three channel npy file
    NOTE: images = tuple containing all three projections
    """
    holder = np.zeros((*output_shape[::-1], 3))
    for i, img in enumerate(images):
        arr = sitk.GetArrayFromImage(img).astype(np.float)
        arr /= 255 # Normalise to [0, 1]
        if 'coronal' in path:
            holder[..., i] = arr.T
        elif 'sagittal' in path:
            holder[..., i] = arr
        else:
            raise ValueError
    os.makedirs(path, exist_ok=True)
    np.save(f'{path}{name}.npy', holder)

def dcm_orientation(file):
    #* Read dcm files and check meta
    #* Determine which dcm series are being read in wrong order 
    reader = sitk.ImageFileReader()
    reader.SetFileName(file)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    k = '0020|0032'
    img_pos = reader.GetMetaData(k)
    x, y, z = (float(x) for x in img_pos.split("\\"))
    return x


def all_equal(iterable):
    #* Check all elems of list are equal
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def vxl_read_dicom(filenames):
    #* List of filenames, sorted in increasing slice number (as vxl does)
    holder = []
    slice_thickness = []
    for file in filenames:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        k = '0018|0050'  # Slice thickness
        thickness = reader.GetMetaData(k)
        slice_thickness.append(thickness)
        img = reader.Execute()
        holder.append(sitk.GetArrayFromImage(img))

    count = Counter(slice_thickness)
    if len(count.keys()) == 2: #! Find calibration slice + remove
        for key, val in count.items():
            if val == 1:
                idx = slice_thickness.index(key)
                del slice_thickness[idx]
                del holder[idx]
        assert all_equal(slice_thickness), f'Mismatch slice thickness'
        slice_shapes = [img.shape for img in holder]
        shape_count = Counter(slice_shapes) 
        #~ Check for inconsistent slice dimensions
        if len(shape_count.keys()) != 1:
            print('Inconsistent in-plane dimensions ')
            return None, None, None

        return np.array(holder), slice_thickness, idx
    else:
        assert all_equal(slice_thickness), f'Mismatch slice thickness'
        slice_shapes = [img.shape for img in holder]
        shape_count = Counter(slice_shapes)
        #~ Check for inconsistent slice dimensions
        if len(shape_count.keys()) != 1:
            print('Inconsistent in-plane dimensions ')
            return None, None, None
        return np.array(holder), slice_thickness, None

#-------- MAIN -------------
def main(min_pix=None, dim=0, plot=False, write=False, output_shape=(512, 512), save_volumes=False):
    # Create lists for values that will be needed later for overlaying annotations
    scale_list = [] # List containing values by which each dimension was scaled when iso. resampling
    padding_list = [] # Values for x and y padding when centering image in output array
    name_list = [] # To keep track of names
    direction_list = [] # To keep track of orientation of initial volume
    origin_list = [] # Keep track of original origin
    orig_pix = [] # Record origin pixel size for coronal annotations 
    orig_thickness = [] #Keep track of slice thickness tag as VXL uses this instead of slice spacing 
    num_slices = [] #Keep track of number of slices to fix issues with slice thickness.
    points = pd.read_csv('./formatted_pts.csv', index_col=0, header=0)

    issue_list = [] # Write filenames with assertion error to file (inconsistent slice thickness)

    for root, dir_, files in os.walk(data_dir):
        # Check if files in directory and only look for sagittal reformats
        split_path = root.split('/')
        if files and '_Sag' in split_path[-1]:
            name = f'{split_path[-3]}_{split_path[-1]}'
            print(name)
        elif files and 'SRS' in split_path[-1]:
            name_split = split_path[-3].split('.') # Split long name according to '.'
            name = f'{split_path[-4]}_{name_split[-1]}_{split_path[-1]}'
            print(name)
        else:
            continue
        reader = sitk.ImageSeriesReader()
        dcm_paths = reader.GetGDCMSeriesFileNames(root)
        
        dcm_names = sorted([path for path in dcm_paths])
        if len(dcm_names) <= 1:
            print('No Dicom files found, skipping directory')
            continue
        
        volume, slice_thickness, calib_idx = vxl_read_dicom(dcm_names) #* calib_idx is index of calibration slice - to remove from dcm names
        if all(v is None for v in [volume, slice_thickness, calib_idx]): #! Check if inconsistent slice dimension
            print('Issue with slice dimensions')
            issue_list.append(name)
            continue

        if calib_idx is not None:
            del dcm_names[calib_idx]
        
        reader.SetFileNames(dcm_names)
        print(f'Reading {name} - {len(dcm_names)} dcm files detected')
        
        #* Get Pixel spacing from 3D volume but read slice by slice (to match VXL)
        meta = reader.Execute()
        direction_list.append(meta.GetDirection())
        origin_list.append(meta.GetOrigin())
        out_img = sitk.GetImageFromArray(np.squeeze(volume))
        out_img.SetSpacing(meta.GetSpacing())
        out_img.SetOrigin(meta.GetOrigin())
        out_img.SetDirection(meta.GetDirection())
        #* Get spacing
        if min_pix is None:
            min_pix = min(meta.GetSpacing())
        new_spacing = (min_pix, min_pix)
        
        #* WL normalisation
        bone_norm_img = WL_norm(out_img, window=1000, level=700)
        tissue_norm_img = WL_norm(out_img, window=600, level=200)

        #* Projections
        mip = maximum_projection(bone_norm_img, dim=dim)
        avg = average_projection(tissue_norm_img, dim=dim)
        avg = normalize(avg)
        std = standard_deviation(tissue_norm_img, dim=dim)
        std = normalize(std)
        #* Resample projections to isotropic voxel size
        if 'coronal' in output_dir:
            padded_mip, scale, padding = post_projection(mip[0], new_spacing, output_shape)
            padded_avg, _,  _ = post_projection(avg[0], new_spacing, output_shape)
            padded_std, _, _ = post_projection(std[0], new_spacing, output_shape)
        elif 'sagittal' in output_dir:
            padded_mip, scale, padding = post_projection(
                mip[:,:,  0], new_spacing, output_shape)
            padded_avg, _,  _ = post_projection(
                avg[:,:, 0], new_spacing, output_shape)
            padded_std, _, _ = post_projection(
                std[:,:, 0], new_spacing, output_shape)
        else:
            raise ValueError

        name_list.append(name)
        padding_list.append(tuple(padding))
        scale_list.append(tuple(scale))
        orig_pix.append(out_img.GetSpacing())
        orig_thickness.append(slice_thickness[0])
        num_slices.append(len(dcm_names))
        if save_volumes:
            #* Write volume to .nii for easier use in notebooks
            new_spacing = (min_pix, min_pix, min_pix)
            
            print(out_img.GetSize(), out_img.GetSpacing(), out_img.GetDirection())
            sitk.WriteImage(
                out_img, output_dir + f'ct_volumes/{name}.nii')

        if plot:
            #* Sanity check plots
            print('Plotting projections')
            plot_projection(padded_mip, name, output_dir + 'mip/') 
            plot_projection(padded_avg, name, output_dir + 'avg/')
            plot_projection(padded_std, name, output_dir + 'std/')
        if write:
            #* Write all projections to numpy (inputs to models)
            print('Writing projections to npy')
            images = (padded_mip, padded_avg, padded_std)
            write_threeChannel(images, name, output_dir + 'all_projections/', output_shape)

    # with open('./dicom_direction.csv', 'w') as f:
    #     print('Writing directions to CSV')
    #     wrt = csv.writer(f, dialect='excel')
    #     wrt.writerow(['Name', 'Direction', 'Origin'])
    #     for name, direction, origin in zip(name_list, direction_list, origin_list):
    #         wrt.writerow([name, direction, origin])

    #Write info needed for annotations to csv file
    with open(output_dir + 'annotation_info.csv', 'w') as f:
        print('Writing CSV File')
        wrt = csv.writer(f, dialect='excel')
        wrt.writerow(['Name', 'Padding', 'Pixel Scaling', 'Orig. Pix', 'Slice Thickness', 'Num Slices'])
        for name, pad, scale, pix, thick, num in zip(name_list, padding_list, scale_list, orig_pix, orig_thickness, num_slices):
            wrt.writerow([name, pad, scale, pix, thick, num])

    with open(output_dir + 'issue_w_slice.csv', 'w') as f:
        wrt = csv.writer(f, dialect='excel')
        wrt.writerow(['Name'])
        for name in issue_list:
            wrt.writerow([name])


if __name__ == '__main__':
    data_dir = '/data/CT_volumes/images/'
    #output_dir = '/data/PAB_data/images_coronal/'  # ! Set dim =0
    output_dir ='./images_sagittal/' #! Set dim = 2
    main(min_pix=4*0.3125, dim=2, plot=True, write=True, save_volumes=False)
