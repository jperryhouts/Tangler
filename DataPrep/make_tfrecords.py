#!/usr/bin/env python3

import math, os, random
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Process
import numpy as np
import tensorflow as tf

def _int_to_feature(value: int) -> tf.train.Feature:
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)

def _array_to_feature(array: np.ndarray, dtype=tf.uint8) -> tf.train.Feature:
    tensor = tf.convert_to_tensor(array, dtype=dtype)
    serialized = tf.io.serialize_tensor(tensor)
    bytes_list = tf.train.BytesList(value=[serialized.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def _bytes_to_feature(values:str) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def files_to_example(image_path:str, target_path:str, n_pins:int=256) -> tf.train.Example:
    raveled = np.loadtxt(target_path, dtype=np.int).flatten()
    if not raveled or len(raveled) == 0:
        print(f">> Unable to load raveled sequence <{target_path}>")
        return None
    if raveled.min() < 0 or raveled.max() >= n_pins:
        print(f">> Value out of bounds in raveled sequence <{target_path}>")
        return None

    image_data = tf.io.gfile.GFile(image_path, 'rb').read()
    image_shape = tf.image.decode_jpeg(image_data).shape
    assert image_shape[0] == image_shape[1], f'{image_shape[0]} != {image_shape[1]} ({image_shape})'
    image_res = image_shape[0]

    feature = {
        'image/encoded': _bytes_to_feature(image_data),
        'image/format': _bytes_to_feature(b'jpeg'),
        'image/res': _int_to_feature(image_res),
        'target/sequence': _array_to_feature(raveled),
        'target/length': _int_to_feature(raveled.size)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

class Shard():
    def __init__(self, path:str) -> None:
        self.examples = []
        self.path = path
        self.basename = os.path.basename(self.path)

    def append(self, img_path:str, target_path:str):
        self.examples.append((img_path, target_path))

    def save(self) -> None:
        print(f">> Saving shard {self.basename} with {len(self.examples)} records.")
        with tf.io.TFRecordWriter(self.fname) as tfrecord_writer:
            for img_path, target_path in self.examples:
                example = files_to_example(img_path, target_path)
                if example is not None:
                    tfrecord_writer.write(example.SerializeToString())
        print(f">> Completed saving {self.basename}")

def write_records(input_dir:str, output_dir:str, num_shards:int=10, sequential:bool=False) -> None:
    assert os.path.isdir(input_dir)
    img_fnames = list(Path(input_dir).rglob('*_norm.jpg'))
    assert len(img_fnames) > 0

    print(f">> Found {len(img_fnames)} examples.")

    random.shuffle(img_fnames)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.isdir(output_dir)

    num_per_shard = int(math.ceil(len(img_fnames)/float(num_shards)))

    shards = []
    for i in range(num_shards):
        tfrecord = f"tangle_{(i+1):05d}-of-{num_shards:05d}.tfrecord"

        shard_images = img_fnames[i*num_per_shard:(i+1)*num_per_shard]
        shard_targets = [img[:-len("_norm.jpg")]+".raveled" for img in shard_images]

        shard = Shard(os.path.join(output_dir, tfrecord))
        for img, target in zip(shard_images, shard_targets):
            if not os.path.isfile(img):
                print(f">> Image not found: {img}")
                continue
            if not os.path.isfile(target):
                print(f">> Target not found: {target}")
                continue
            shard.append(img, target)

        shards.append(shard)

    if sequential:
        for shard in shards:
            shard.save()
    else:
        procs = [Process(target=shard.save) for shard in shards]

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num-shards', '-n', type=int, default=10,
        help="Split dataset into multiple files (default: 10)")
    parser.add_argument('--sequential', action='store_true',
        help='Process tfrecords one at a time (defaults to each tfrecord file in parallel process)')
    parser.add_argument('input', help='Root directory for dataset (e.g. ./train)')
    parser.add_argument('output', help='Directory in which to save tfrecord files')
    args = parser.parse_args()

    write_records(args.input, args.output, args.num_shards, args.sequential)