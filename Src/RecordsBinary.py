#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import argparse
from segmentation_net import create_tfrecord, compute_mean


from datagen_object import DataGenNeerajBin, ExampleUNetDatagen



def ArgumentOptions():
    parser = argparse.ArgumentParser(
        description='TFRecord options')
    parser.add_argument('--data1', required=True,
                        metavar="/path/to/dataset1/",
                        help='Root directory of the dataset1')
    parser.add_argument('--data2', required=True,
                        metavar="/path/to/dataset2/",
                        help='Root directory of the dataset2')
    parser.add_argument('--test', required=True,
                        metavar="/path/to/dataset_fortest/",
                        help='Root directory of the test set')
    parser.add_argument('--output_train', required=True,
                        metavar="string",
                        help="String for the output train record")
    parser.add_argument('--output_test', required=True,
                        metavar="string",
                        help="String for the output train record")
    parser.add_argument('--output_mean_array', required=True,
                        metavar="string",
                        help="String for the output mean_array")
    args = parser.parse_args()
    return args

def main():
    args_ = ArgumentOptions()
    glob_data1 = os.path.join(args_.data1, "Slide_*", "*.png")
    glob_data2 = os.path.join(args_.data2, "Slide_*", "*.png")
    glob_test = os.path.join(args_.test, "Slide_*", "*.png")
    dg2 = DataGenNeerajBin(glob_data2, crop=16, verbose=True, cache=True)  # --data2 $neeraj
    dg1 = ExampleUNetDatagen(glob_data1, verbose=True, cache=True)
    dg_test = ExampleUNetDatagen(glob_test, verbose=True, cache=True)

    create_tfrecord(args_.output_train, [dg1, dg2])
    create_tfrecord(args_.output_test, [dg_test])
    compute_mean(args_.output_mean_array, [dg_test, dg1, dg2])

if __name__ == '__main__':
    main()
