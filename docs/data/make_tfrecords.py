#!/usr/bin/env python3

import math, os, random, logging
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Process
from typing import Iterable
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from PIL import Image
    plt.style.use('seaborn-talk')
except Exception as e:
    print(e)

import tensorflow as tf

## This script is not GPU intensive, but tensorflow likes to allocate
## GPU RAM anyway. Disable GPU access to avoid hogging that resource.
tf.config.set_visible_devices([], 'GPU')

_N_PINS = 256
_RECORD_FORMAT_VERSION = 6

def _int_to_feature(value: int) -> tf.train.Feature:
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)

def _tensor_to_feature(tensor: tf.Tensor) -> tf.train.Feature:
    serialized = tf.io.serialize_tensor(tensor)
    bytes_list = tf.train.BytesList(value=[serialized.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def _bytes_to_feature(values:str) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def pin_path_to_target(path:Iterable) -> np.ndarray:
    target = np.zeros((_N_PINS, _N_PINS), dtype=np.uint8)
    target[path[:-1],path[1:]] = 1
    target[path[1:],path[:-1]] = 1
    return target

def load_raveled(path:str) -> np.ndarray:
    raveled = np.loadtxt(path, dtype=np.int).flatten()
    if raveled is None or len(raveled) == 0:
        logging.warn(f">> Unable to load raveled sequence <{path}>")
        return None
    if raveled.min() < 0 or raveled.max() >= _N_PINS:
        logging.warn(f">> Value out of bounds in raveled sequence <{path}>")
        return None
    return raveled

def visualize(raveled_path:str) -> None:
    img_path = str(raveled_path).rsplit('.')[0]+'.jpg'
    img = Image.open(img_path)
    img.resize((256,256))

    raveled = load_raveled(raveled_path)
    target = pin_path_to_target(raveled)

    _, ax = plt.subplots(1,2, figsize=(9.5,4))

    ax[0].imshow(img, cmap=plt.cm.gray, aspect=1)
    print(target.max())
    tp = ax[1].imshow(1+target, cmap=plt.cm.magma, aspect=1, interpolation='nearest',
        norm=colors.LogNorm(vmin=1, vmax=target.max()+1))
    cbar = plt.colorbar(tp, ax=ax[1], ticks=[1, 3, 5, 9, 17])
    cbar.ax.set_yticklabels(['0', '2', '4', '8', '16'])
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.show()

def files_to_tfexample(base_path:str, res:int=-1) -> tf.train.Example:
    raveled = load_raveled(f"{base_path}.raveled")
    if raveled is None:
        return None

    image_data = tf.io.gfile.GFile(f"{base_path}.jpg", 'rb').read()
    image = tf.io.decode_jpeg(image_data)

    if res != -1 and (image.shape[0] != res or image.shape[1] != res):
        image = tf.image.resize(image, (res, res), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_data = tf.io.encode_jpeg(image).numpy()
        image_res = res

    assert image.shape[0] == image.shape[1], f'Invalid image shape: {image.shape}'
    image_res = image.shape[0]

    # Save sparse representation of target tensor (i.e. only store the
    # indices of non-zero elements)
    target = pin_path_to_target(raveled)
    target = tf.convert_to_tensor(np.tril(target), tf.uint8)
    sp_target = tf.sparse.from_dense(target)
    sp_indices = tf.cast(sp_target.indices, tf.uint8)

    return tf.train.Example(features=tf.train.Features(feature={
        'record/version': _int_to_feature(_RECORD_FORMAT_VERSION),
        'record/name': _bytes_to_feature(base_path.encode('utf-8')),
        'image/encoded': _bytes_to_feature(image_data),
        'image/res': _int_to_feature(image_res),
        'target/n_pins': _int_to_feature(_N_PINS),
        'target/sparsity': _int_to_feature(sp_indices.shape[0]),
        'target/indices': _tensor_to_feature(tf.reshape(sp_indices, [-1])),
    }))

class Shard():
    def __init__(self, path:str, res:int) -> None:
        self.examples = []
        self.path = path
        self.res = res
        self.basename = os.path.basename(self.path)

    def append(self, stem:str):
        self.examples.append(stem)

    def save(self) -> None:
        print(f">> Saving shard {self.basename} with {len(self.examples)} records.")
        with tf.io.TFRecordWriter(self.path) as tfrecord_writer:
            for stem in self.examples:
                try:
                    example = files_to_tfexample(stem, self.res)
                    if example is not None:
                        tfrecord_writer.write(example.SerializeToString())
                except Exception as e:
                    logging.error(f"Unable to load example <{stem}>",e)
        print(f">> Completed saving {self.basename}")

def write_records(input_dir:str, output_dir:str, num_shards:int=10, sequential:bool=False, res:int=-1) -> None:
    assert os.path.isdir(input_dir)
    get_stem = lambda p: os.path.join(p.parent.absolute(), p.stem)
    examples = [get_stem(path) for path in Path(input_dir).rglob('*.jpg')]
    assert len(examples) > 0

    print(f">> Found {len(examples)} examples.")

    random.shuffle(examples)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.isdir(output_dir)

    num_per_shard = int(math.ceil(len(examples)/float(num_shards)))

    shards = []
    for i in range(num_shards):
        tfrecord = f"tangle_{(i+1):05d}-of-{num_shards:05d}.tfrecord"
        shard = Shard(os.path.join(output_dir, tfrecord), res)
        for stem in examples[i*num_per_shard:(i+1)*num_per_shard]:
            img = f"{stem}.jpg"
            if not os.path.isfile(img):
                print(f">> Image not found: {img}")
                continue
            target = f"{stem}.raveled"
            if not os.path.isfile(target):
                print(f">> Target not found: {target}")
                continue
            shard.append(stem)

        shards.append(shard)

    if sequential or len(shards) == 1:
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
        help='Process tfrecords one at a time (defaults to each tfrecord file in a separate process)')
    parser.add_argument('--resolution', '-r', type=int, default=-1,
        help='Resize image to this many pixels per side. Defaults to no resizing')
    parser.add_argument('input', help='Root directory for dataset (e.g. ./train)')
    parser.add_argument('output', help='Directory in which to save tfrecord files. Use "@vis" to display data')
    args = parser.parse_args()

    if args.output == "@vis":
        targets = list(Path(args.input).rglob('*.raveled'))[:10]
        random.shuffle(targets)
        for raveled_path in targets:
            visualize(raveled_path)

    write_records(args.input, args.output, args.num_shards, args.sequential, args.resolution)