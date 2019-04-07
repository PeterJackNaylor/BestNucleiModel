
import os
import numpy as np
from operator import add

from scipy.ndimage import binary_fill_holes

from dynamic_watershed.dynamic_watershed import post_process, generate_wsl
from skimage.morphology import remove_small_objects
import skimage.measure as meas
from sklearn.metrics import confusion_matrix
from skimage import img_as_ubyte
from skimage.segmentation import find_boundaries, mark_boundaries
from skimage.color import hsv2rgb, label2rgb

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


def AJI_fast(G, S):
    """
    AJI as described in the paper, but a much faster implementation.
    """
    G = meas.label(G, background=0)
    S = meas.label(S, background=0)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0 
    USED = np.zeros(S.max())

    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()
        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]

            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union
        
        JI_ligne = list(map(h, range(1, S_max + 1)))
        best_indice = np.argmax(JI_ligne) + 1
        C += cm[i, best_indice]
        U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
        USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U) 

def f(tup):
    return np.expand_dims(np.expand_dims(np.array(tup, dtype='float'),0), 0)

def inv_f(pix):
    return np.squeeze(pix)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(float(i) / N, 1 , brightness) for i in range(N)]
    colors = list(map(lambda c: inv_f(hsv2rgb(f(c))), hsv))
    np.random.shuffle(colors)
    colors = np.array(colors)
    n, p = colors.shape
    new_colors = np.zeros(shape=(n+1, p))
    new_colors[1:, :] = colors
    return new_colors

def add_contours(image, label, color = (0, 1, 0)):
    
    # mask = find_boundaries(label)
    # res = np.array(image).copy()
    # res[mask] = np.array([0, 255, 0])
    res = mark_boundaries(image, label, color=color)
    res = img_as_ubyte(res)
    return res

def apply_mask_with_highlighted_borders(image, labeled, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for i in range(1, labeled.max() + 1):
        for c in range(3):
            image = add_contours(image, labeled == i, color = color[i])
            image[:, :, c] = np.where(labeled == i,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[i, c] * 255,
                                      image[:, :, c])
    return image
