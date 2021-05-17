#!/usr/bin/env python3

from .utils import load_img

import glob, os
import math
from argparse import ArgumentParser

import numpy as np

import tensorflow as tf

IMG_RES = 600
N_PINS = 300

def _int_to_feature(value: int) -> tf.train.Feature:
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)

def _tensor_to_feature(tensor: tf.Tensor) -> tf.train.Feature:
    serialized = tf.io.serialize_tensor(tensor)
    bytes_list = tf.train.BytesList(value=[serialized.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def load_example(stem: str, res:int) -> tf.train.Example:
    pixels = load_img(stem+'.JPEG', res)
    path = np.loadtxt(stem+'.tsv')

    t_pixels = tf.convert_to_tensor(pixels.reshape((res,res,1)), dtype=tf.float32)
    t_path = tf.convert_to_tensor(path.flatten(), dtype=tf.float32)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/pixels': _tensor_to_feature(t_pixels),
        'image/size': _int_to_feature(res),
        'target': _tensor_to_feature(t_path)
    }))

def writeTFRecords(input_dir:str, output_dir:str, num_shards:int, res:int) -> None:
    fnames = glob.glob(os.path.join(input_dir, '*', '*.JPEG'))

    n_images = len(fnames)
    num_per_shard = int(math.ceil(n_images/float(num_shards)))

    for shard_id in range(num_shards):
        output_basename = f"res={res}_{(shard_id+1):05d}-of-{num_shards:05d}.tfrecord"
        output_path = os.path.join(output_dir, output_basename)

        with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min(start_idx+num_per_shard, n_images)
            for i in range(start_idx, end_idx):
                img_fname = fnames[i]
                stem = img_fname.rsplit('.', 1)[0]
                example = load_example(stem, res)
                tfrecord_writer.write(example.SerializeToString())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--resolution', '-r', type=int, default=600,
        help="Crop/scale images to squares with this number of pixels per side (default: 600)")
    parser.add_argument('--num-shards', '-n', type=int, default=10,
        help="Split dataset into this many tfrecord files (default: 10)")
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    assert os.path.isdir(args.input)
    assert not os.path.exists(args.output)

    os.makedirs(args.output)

    writeTFRecords(args.input, args.output, num_shards=args.num_shards, res=args.resolution)