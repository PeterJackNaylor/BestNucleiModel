
import os
from glob import glob
from skimage.io import imread, imsave
import skimage.measure as meas
from numpy import load
from segmentation_net import DistanceUnet
from utils import expend, PostProcessOut, AJI_fast, check_or_create
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
    parser.add_argument('--output_path', required=True,
                        metavar="float", type=str,
                        help='output')
    args = parser.parse_args()
    return args

def resize(tup):
    rgb, lbl = tup 
    rgb = expend(rgb[0:500, 0:500,0:3], 92, 92)
    lbl = lbl[0:500, 0:500]
    return rgb, lbl

def load_data(f):
    rgb = imread(f)
    label = imread(f.replace("Slide", "GT"))
    return rgb, label


def test_model(folderpath, model, output):
    scores = {"f1": [],
              "aji": []}
    files = glob(os.path.join(folderpath, "Slide_*", "*.png"))
    check_or_create(output)
    num = 0
    for f in files:
        rgb, label = resize(load_data(f))
        dic_res = model.predict(rgb, label=label)
        label = meas.label(label)

        f1 = dic_res['f1_score']
        label_int = PostProcessOut(dic_res['probability'][:,:,0])
        aji = AJI_fast(label, label_int)

        import pdb; pdb.set_trace()
        scores["f1"].append(f1)
        scores["aji"].append(aji)
        colors = random_colors(255)
        output_gt = apply_mask_with_highlighted_borders(rgb, label, color, alpha=0.5)
        output = apply_mask_with_highlighted_borders(rgb, label_int, color, alpha=0.5)
        num += 1
        imsave(os.path.join(output, "test_{}_gt.png"), output_gt)
        imsave(os.path.join(output, "test_{}_pred.png"), output)
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

    res = test_model(args.test_folder, model, options.output_path)

    model.sess.close() 
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
