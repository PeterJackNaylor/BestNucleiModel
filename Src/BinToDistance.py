from skimage.io import imread, imsave
from glob import glob
from os.path import dirname, join, basename
from shutil import copy
import os
from scipy.ndimage.morphology import distance_transform_cdt
import numpy as np
import sys

def LoadGT(path):
    img = imread(path, dtype='uint8')
    return img

def DistanceWithoutNormalise(bin_image):
    res = np.zeros_like(bin_image)
    for j in range(1, bin_image.max() + 1):
        one_cell = np.zeros_like(bin_image)
        one_cell[bin_image == j] = 1
        one_cell = distance_transform_cdt(one_cell)
        res[bin_image == j] = one_cell[bin_image == j]
    res = res.astype('uint8')
    return res

def CheckOrCreate(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


NEW_FOLDER = sys.argv[2]
CheckOrCreate(NEW_FOLDER)

for image in glob('{}/Slide_*/*.png'.format(sys.argv[1])):
    baseN = basename(image)
    Slide_name = dirname(image)
    GT_name = baseN.replace('Slide', 'GT')
    OLD_FOLDER = dirname(Slide_name)
    Slide_N = basename(dirname(image))
    GT_N = Slide_N.replace('Slide_', 'GT_')
    
    CheckOrCreate(join(NEW_FOLDER, Slide_N))
    CheckOrCreate(join(NEW_FOLDER, GT_N))

    copy(image, join(NEW_FOLDER, Slide_N, baseN))
    bin_image = LoadGT(join(OLD_FOLDER, GT_N, GT_name))
    res = DistanceWithoutNormalise(bin_image)
imsave(join(NEW_FOLDER, GT_N, GT_name), res)