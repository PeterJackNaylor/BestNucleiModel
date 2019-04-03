
from numpy import load
from segmentation_net import DistanceUnet
from utils import expend

def GetOptions():
    import argparse
    parser = argparse.ArgumentParser(
        description='Training Distance')
    parser.add_argument('--model', required=True,
                        metavar="str", type=str,
                        help='model weights')
    parser.add_argument('--features', required=True,
                        metavar="str", type=int,
                        help='Complexity of the architecture in terms of filters')
    parser.add_argument('--mean_array', required=True,
                        metavar="float", type=str,
                        help='path to the mean_file')
    parser.add_argument('--test_folder', required=True,
                        metavar="float", type=str,
                        help='path to the test folder')
    args = parser.parse_args()
    return args
import os
from glob import glob
from skimage.io import imload


def resize(tup):
    rgb, lbl = tup 
    rgb = expend(rgb[0:504], 92, 92)
    lbl = lbl[0:504]
    return rgb, lbl

def load_data(f):
    rgb = imload(rgb)
    label = imload(rgb.replace("Slide", "GT"))
    return rgb, label


def test_model(folderpath, model):
    scores = {"f1": []
              "acc": []}
    files = glob(os.path.join(folderpath, "Slide_*", "*.png"))
    
    for f in files:
        rgb, label = resize(load_data(f))
        dic_res = model.predict(rgb, label=label)
        import pdb; pdb.set_trace()

def main():

    args = GetOptions()

    variables_model = {
        ## Model basics
        "num_labels": 2,
        "image_size": (212, 212),
        "log": args.model, 
        "num_channels": 3,
        "tensorboard": False,
        "seed": None, 
        "verbose": 1,
        "n_features": args.features
    }

    model = DistanceUnet(**variables_model)


    res = test_model(args.test_folder, model)

    model.sess.close() 
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
