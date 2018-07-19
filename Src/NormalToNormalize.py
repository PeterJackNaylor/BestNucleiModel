import sys

from skimage.io import imread, imsave
from glob import glob
from os.path import dirname, join, basename
from shutil import copy
import os
from staintools.utils.visual import read_image
from staintools import ReinhardNormalizer, MacenkoNormalizer, VahadaneNormalizer


def PrepNormalizer(normalization, targetPath):
    if "Reinhard" == normalization:
        n = ReinhardNormalizer()
    elif "Macenko" == normalization:
        n = MacenkoNormalizer()
    elif "Vahadane" == normalization:
        n = VahadaneNormalizer()
    else:
        print("No knowned normalization given..")
    targetImg = read_image(targetPath)
    n.fit(targetImg)
    return n



def LoadRGB(path):
    img = imread(path, dtype='uint8')[:,:,0:3]
    return img

def Normalise(img, ref_img):

    n = PrepNormalizer("Vahadane", ref_img)
    res = n.transform(img)

    return res

def CheckOrCreate(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


NEW_FOLDER = sys.argv[2]
CheckOrCreate(NEW_FOLDER)
ref_image = sys.argv[3]

for image in glob('{}/Slide_*/*.png'.format(sys.argv[1])):
    baseN = basename(image)
    Slide_name = dirname(image)
    GT_name = baseN.replace('Slide', 'GT')
    OLD_FOLDER = dirname(Slide_name)
    Slide_N = basename(dirname(image))
    GT_N = Slide_N.replace('Slide_', 'GT_')
    CheckOrCreate(join(NEW_FOLDER, Slide_N))
    CheckOrCreate(join(NEW_FOLDER, GT_N))

    copy(join(OLD_FOLDER, GT_N, GT_name), join(NEW_FOLDER, GT_N, GT_name))

    rgb_image = LoadRGB(image)
    res = Normalise(rgb_image, ref_image)
    imsave(join(NEW_FOLDER, Slide_N, baseN), res)