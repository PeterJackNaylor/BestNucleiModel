
import os
from glob import glob
from skimage.io import imread
import skimage.measure as meas
from numpy import load
from segmentation_net import DistanceUnet
from utils import expend, PostProcessOut

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

def resize(tup):
    rgb, lbl = tup 
    rgb = expend(rgb[0:504, 0:504,0:2], 92, 92)
    lbl = lbl[0:504, 0:504]
    return rgb, lbl

def load_data(f):
    rgb = imread(f)
    label = imread(rgb.replace("Slide", "GT"))
    return rgb, label


def test_model(folderpath, model):
    scores = {"f1": [],
              "aji": []}
    files = glob(os.path.join(folderpath, "Slide_*", "*.png"))
    
    for f in files:
        rgb, label = resize(load_data(f))
        dic_res = model.predict(rgb, label=label)
        f1 = dic_res['f1_score']
        label_int = PostProcessOut(dic_res['probability'])
        label = meas.label(label)

        import pdb; pdb.set_trace()

def main():

    args = GetOptions()

    variables_model = {
        ## Model basics

        "image_size": (212, 212),
        "log": args.model, 
        "num_channels": 3,
        # "num_labels": 2, #remove from distance
        'mean_array': load(args.mean_array),
        "seed": None, 
        "verbose": 1,
        "fake_batch": 1,
        "n_features": args.features
    }

    model = DistanceUnet(**variables_model)

    res = test_model(args.test_folder, model)

    model.sess.close() 
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
