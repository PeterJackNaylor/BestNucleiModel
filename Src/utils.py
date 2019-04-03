
import os
import numpy as np
from operator import add

from scipy.ndimage import binary_fill_holes

from dynamic_watershed.dynamic_watershed import post_process, generate_wsl
from skimage.morphology import remove_small_objects
import skimage.measure as meas


def check_or_create(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def flip_vertical(picture):
    """ 
    vertical flip
    takes an arbitrary image as entry
    """
    res = np.flip(picture, axis=1)
    return res


def flip_horizontal(picture):
    """
    horizontal flip
    takes an arbitrary image as entry
    """
    res = np.flip(picture, axis=0)
    return res

def expend(image, x_s, y_s):
    """
    Expend the image and mirror the border of size x and y
    """
    rows, cols = image.shape[0], image.shape[1]
    if len(image.shape) == 2:
        enlarged_image = np.zeros(shape=(rows + 2*y_s, cols + 2*x_s))
    else:
        enlarged_image = np.zeros(shape=(rows + 2*y_s, cols + 2*x_s, 3))

    enlarged_image[y_s:(y_s + rows), x_s:(x_s + cols)] = image
    # top part:
    enlarged_image[0:y_s, x_s:(x_s + cols)] = flip_horizontal(
        enlarged_image[y_s:(2*y_s), x_s:(x_s + cols)])
    # bottom part:
    enlarged_image[(y_s + rows):(2*y_s + rows), x_s:(x_s + cols)] = flip_horizontal(
        enlarged_image[rows:(y_s + rows), x_s:(x_s + cols)])
    # left part:
    enlarged_image[y_s:(y_s + rows), 0:x_s] = flip_vertical(
        enlarged_image[y_s:(y_s + rows), x_s:(2*x_s)])
    # right part:
    enlarged_image[y_s:(y_s + rows), (cols + x_s):(2*x_s + cols)] = flip_vertical(
        enlarged_image[y_s:(y_s + rows), cols:(cols + x_s)])
    # top left from left part:
    enlarged_image[0:y_s, 0:x_s] = flip_horizontal(
        enlarged_image[y_s:(2*y_s), 0:x_s])
    # top right from right part:
    enlarged_image[0:y_s, (x_s + cols):(2*x_s + cols)] = flip_horizontal(
        enlarged_image[y_s:(2*y_s), cols:(x_s + cols)])
    # bottom left from left part:
    enlarged_image[(y_s + rows):(2*y_s + rows), 0:x_s] = flip_horizontal(
        enlarged_image[rows:(y_s + rows), 0:x_s])
    # bottom right from right part
    enlarged_image[(y_s + rows):(2*y_s + rows), (x_s + cols):(2*x_s + cols)] = flip_horizontal(
        enlarged_image[rows:(y_s + rows), (x_s + cols):(2*x_s + cols)])
    enlarged_image = enlarged_image.astype('uint8')
    return enlarged_image

def fill_holes(image):
    rec = binary_fill_holes(image)
    return rec


def PostProcessOut(pred):
    hp = {'p1': 1, 'p2':0.5}
    pred[pred < 0] = 0.
    pred[pred > 255] = 255.
    labeled_pic = post_process(pred, hp["p1"], hp["p2"])

    borders_labeled_pic = generate_wsl(labeled_pic)
    min_size = 128
    labeled_pic = remove_small_objects(labeled_pic, min_size=min_size)
    labeled_pic[labeled_pic > 0] = 255
    labeled_pic[borders_labeled_pic > 0] = 0
    labeled_pic = fill_holes(labeled_pic)
    labeled_pic = meas.label(labeled_pic)
    return labeled_pic

