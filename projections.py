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
    print(sitk.GetArrayFromImage(img).max())
    out = wl.Execute(img)
    # out -= minval
    # out /= window
    print(sitk.GetArrayFromImage(out).max())
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
    print(img.GetSize(), output_shape)
    if img.GetSize() != output_shape:
        resampled_img, scale = resample(img, new_spacing)
        padded_img, padding = pad_image(resampled_img, output_shape)
        return padded_img, scale, padding
    else:
        return img, (1, 1), (0, 0)

def plot_projection(img, name, path):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('off')
    arr = sitk.GetArrayFromImage(img)
    print(path, arr.max(), arr.min())
    im = Image.fromarray(arr.astype(np.uint8))
    im.save(f'{path}{name}.png')
    #im.save('test.png')
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
        holder[..., i] = arr
    np.save(f'{path}{name}.npy', holder)

#-------- MAIN -------------
def main(min_pix=None, dim=0, plot=False, write=False, output_shape=(512, 512), save_volumes=False):
    # Create lists for values that will be needed later for overlaying annotations
    scale_list = [] # List containing values by which each dimension was scaled when iso. resampling
    padding_list = [] # Values for x and y padding when centering image in output array
    name_list = [] # To keep track of names
    points = pd.read_csv('./formatted_pts.csv', names=ordered_verts, header=0)
    for root, dir_, files in os.walk(data_dir):
        # Check if files in directory and only look for sagittal reformats
        if files and '_Sag' in root:
            try:
                reader = sitk.ImageSeriesReader()
                dcm_paths = reader.GetGDCMSeriesFileNames(root)
                # Remove tester slice
                dcm_names = [path for path in dcm_paths if not path.endswith('00001.dcm')]
                if len(dcm_names) == 0:
                    print('No Dicom files found, skipping directory')
                    continue
                split_path = root.split('/')
                name = f'{split_path[-3]}_{split_path[-1]}'

                if name != '01_06_2014_363_Sag':
                    print('Skipping', name)
                    continue
                reader.SetFileNames(list(dcm_names))
                print(f'Reading {name} - {len(dcm_names)} dcm files detected')
                volume = reader.Execute()
                if min_pix is None:
                    min_pix = min(img.GetSpacing())
                # Paul's data needs rotating
                data_block = sitk.GetArrayViewFromImage(volume)
                data_block = np.rot90(np.flip(data_block, axis=0), k=3, axes=(0, 1))
                
                img = sitk.GetImageFromArray(data_block)
                # SITK reorders dimensions so need to account for this when defining spacing
                orig_spacing = list(volume.GetSpacing())
                new_order = [0, 2, 1]
                ordered_spacing = [orig_spacing[i] for i in new_order]
                img.SetSpacing(ordered_spacing)
                new_spacing = (min_pix, min_pix)
                
                # WL normalisation
                bone_norm_img = WL_norm(img, window=1000, level=700)
                tissue_norm_img = WL_norm(img, window=600, level=200)

                print(img.GetSize(), img.GetSpacing())
                # Projections
                mip = maximum_projection(bone_norm_img, dim=dim)
                avg = average_projection(tissue_norm_img, dim=dim)
                avg = normalize(avg)
                std = standard_deviation(tissue_norm_img, dim=dim)
                std = normalize(std)
                # Resample projections to isotropic voxel size
                padded_mip, scale, padding = post_projection(mip[0], new_spacing, output_shape)
                padded_avg, _,  _ = post_projection(avg[0], new_spacing, output_shape)
                padded_std, _, _ = post_projection(std[0], new_spacing, output_shape)

                name_list.append(name)
                padding_list.append(tuple(padding))
                scale_list.append(tuple(scale))
                
                if save_volumes:
                    if f'{name}_kj' in points.index.to_list():
                        new_spacing = (min_pix, min_pix, min_pix)
                        print(img.GetSize())
                        resampled_img, scaling = resample(img, new_spacing)
                        resampled_data = sitk.GetArrayFromImage(resampled_img)
                        print(resampled_data.shape)
                        #get_midline(resampled_data, name, points)

                        # ! Write CT volume to nii for easy use
                        sitk.WriteImage(
                            img, f'./ct_volumes/{name}.nii')
                        break
                    else:
                        print("Can't find name in points df")
                        continue

                if plot:
                    print('Plotting projections')
                    plot_projection(padded_mip, name, output_dir + 'mip/')
                    plot_projection(padded_avg, name, output_dir + 'avg/')
                    plot_projection(padded_std, name, output_dir + 'std/')
                if write:
                    print('Writing projections to npy')
                    images = (padded_mip, padded_avg, padded_std)
                    write_threeChannel(images, name, output_dir + 'all_projections/', output_shape)
        
            except RuntimeError:
                continue
    #Write info needed for annotations to csv file
    # with open(output_dir + 'annotation_info.csv', 'w') as f:
    #     print('Writing CSV File')
    #     wrt = csv.writer(f, dialect='excel')
    #     wrt.writerow(['Name', 'Padding', 'Pixel Scaling'])
    #     for name, pad, scale in zip(name_list, padding_list, scale_list):
    #         wrt.writerow([name, pad, scale])



data_dir = '/home/donal/CT_volumes/images/'
output_dir = './images_coronal/'
#output_dir ='./images_sagittal/'

if __name__ == '__main__':
    main(min_pix=4*0.3125, dim=0, plot=False, write=False, save_volumes=True)
