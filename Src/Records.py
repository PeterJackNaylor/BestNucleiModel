#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import numpy as np
from DataObject.DataGenClass import DataGenMulti
from DataObject.ImageTransform import ListTransform

def ArgumentOptions():
    parser = argparse.ArgumentParser(
        description='TFRecord options')
    parser.add_argument('--test_size', required=True,
                        metavar="integer", type=int,
                        help='Number of samples in the test size')
    parser.add_argument('--data1', required=True,
                        metavar="/path/to/dataset1/",
                        help='Root directory of the dataset1')
    parser.add_argument('--data2', required=True,
                        metavar="/path/to/dataset2/",
                        help='Root directory of the dataset2')
    parser.add_argument('--output_train', required=True,
                        metavar="string",
                        help="String for the output train record")
    parser.add_argument('--output_test', required=True,
                        metavar="string",
                        help="String for the output train record")
    args = parser.parse_args()
    return args

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def CreateTFRecord(OUTNAME, LIST_DOUBLE):

    tfrecords_filename = OUTNAME
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for img, annotation in LIST_DOUBLE:
#        img = img.astype(np.uint8)
        annotation = annotation.astype(np.uint8)
        height = img.shape[0]
        width = img.shape[1]
   
        img_raw = img.tostring()
        annotation_raw = annotation.tostring()
      
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))
      
        writer.write(example.SerializeToString())
    writer.close()

def fetch_data(path, split, crop, test_patient = ['bougliglou']):
    transform_list, transform_list_test = ListTransform()
    UNET = True
    DG = DataGenMulti(path, split=split, crop = crop, size=(212, 212), 
                      transforms=transform_list_test, UNet=UNET, num=test_patient,
                      mean_file=None, seed_=42)
    original_images = []
    key = DG.RandomKey(False)
    for _ in range(DG.length):
        key = DG.NextKeyRandList(0)
        img, annotation = DG[key]
        original_images.append((img, annotation))
    return original_images

def fetch_data1(path, split="train", test_patient=['bougliglou']):
    """
    Fetch data peter
    """
    return fetch_data(path, split, 4, test_patient)

def fetch_data2(path, split="train", test_patient=['bougliglou']):
    """
    Fetch data neeraj
    """
    return fetch_data(path, split, 16,test_patient)


if __name__ == '__main__':
    args = ArgumentOptions()
    list_data1 = fetch_data1(args.data1)
    list_data2 = fetch_data2(args.data2)
    list_data = list_data1 + list_data2
    ind_test = np.random.choice(len(list_data), size=args.test_size, replace=False)
    ind_train = np.array([i for i in range(len(list_data)) if i not in ind_test])
    CreateTFRecord(args.output_train, list(np.array(list_data)[ind_train]))
    CreateTFRecord(args.output_test , list(np.array(list_data)[ind_train]))