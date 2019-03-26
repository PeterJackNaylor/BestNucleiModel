
import os
import numpy as np
from segmentation_net import ExampleUNetDatagen, ExampleDistDG
from skimage.io import imread

def return_range(inti):
    if inti == 0:
        min_, max_ = 0, 250
    elif inti == 1:
        min_, max_ = 250, 500
    elif inti == 2:
        min_, max_ = 500, 750
    elif inti == 3:
        min_, max_ = 750, 1000
    return min_, max_


def return_rangeCPM(inti, shape, add):
    if inti == 0:
        min_, max_ = 0, (shape - add) // 2 
    elif inti == 1:
        min_, max_ = (shape - add) // 2, (shape - add)
    else:
        raise ValueError("not possible")
    return min_, max_


class DataGenNeerajBin(ExampleUNetDatagen):
    def quarter_image(self, image, integer):
        """
        Split images in quarters dependings of integer
        with additionnal expanding
        """
        add2 = self.add * 2

        row = integer % (self.crop**0.5)
        col = integer // (self.crop**0.5)

        x_b, x_e = return_range(row)
        y_b, y_e = return_range(col)
        
        x_e += add2
        y_e += add2

        return image[x_b:x_e, y_b:y_e]

class DataToyExample(ExampleUNetDatagen):
    def load_mask(self, image_name):
        """
        Way of loading mask images
        """
        mask_name = image_name.replace('Slide', 'GT')
        mask = imread(mask_name)
        return mask


class DataGenNeeraj(ExampleDistDG):
    def quarter_image(self, image, integer):
        """
        Split images in quarters dependings of integer
        with additionnal expanding
        """
        add2 = self.add * 2

        row = integer % (self.crop**0.5)
        col = integer // (self.crop**0.5)

        x_b, x_e = return_range(row)
        y_b, y_e = return_range(col)
        
        x_e += add2
        y_e += add2

        return image[x_b:x_e, y_b:y_e]

def load_mask_from_txt_cpm(path):

    f = open(path, "r")
    lines = f.readlines()
    x, y = int(lines[0].split(" ")[0]), int(lines[0].split(" ")[-1].split('\\')[0])
    del lines[0]
    lines = [int(el.split('\\')[0]) for el in lines]
    lines = np.array(lines)
    if "CPM18" in path:
        mask = lines.reshape(y, x)
    else:
        mask = lines.reshape(x, y)

    return mask


class DataGenCPM(ExampleDistDG):
    def quarter_image(self, image, integer):
        """
        Split images in quarters dependings of integer
        with additionnal expanding
        """
        add2 = self.add * 2        
        row = integer % (self.crop**0.5)
        col = integer // (self.crop**0.5)
        shape = image.shape

        x_b, x_e = return_rangeCPM(row, shape[0], add2)
        y_b, y_e = return_rangeCPM(col, shape[1], add2)
        
        x_e += add2
        y_e += add2

        return image[x_b:x_e, y_b:y_e]

    def load_mask(self, image_name):
        """
        Way of loading mask images
        """

        base = os.path.basename(image_name)
        base = base.replace(".png", "_mask.txt")
        base = base.replace("Slide_", "image")
        path = os.path.join(*image_name.split('/')[:-1])
        path = path.replace("Slide", "GT")
        mask_path = os.path.join(path, base)
        mask = load_mask_from_txt_cpm(mask_path)

        return mask

