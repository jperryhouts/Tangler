#!/usr/bin/env python3

import math, os, random
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Process
from typing import Iterable
import numpy as np

try:
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-talk')
except Exception as e:
    pass

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

_N_CONS = 64
_N_PINS = 256
_RECORD_FORMAT_VERSION = 3

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
    NC2 = _N_CONS//2

    path = path.astype(np.int)
    setA = [set() for _ in range(_N_PINS)]
    setB = [set() for _ in range(_N_PINS)]

    for i in range(len(path)-1):
        a, b = path[i:i+2]
        if len(setA[a]) < NC2:
            setA[a].add((b-a)%_N_PINS)
        if len(setB[b]) < NC2:
            setB[b].add((a-b)%_N_PINS)

    target = np.zeros((_N_PINS, _N_CONS), dtype=np.uint8)
    for p in range(_N_PINS):
        target[p][NC2-len(setA[p]):NC2] = np.array(sorted(setA[p], reverse=False), dtype=np.uint8)
        target[p][NC2:NC2+len(setB[p])] = np.array(sorted(setB[p], reverse=True), dtype=np.uint8)

    return target

def load_raveled(path:str) -> np.ndarray:
    raveled = np.loadtxt(path, dtype=np.int).flatten()
    if raveled is None or len(raveled) == 0:
        print(f">> Unable to load raveled sequence <{path}>")
        return None
    if raveled.min() < 0 or raveled.max() >= _N_PINS:
        print(f">> Value out of bounds in raveled sequence <{path}>")
        return None

    return raveled

def visualize(raveled_path:str) -> None:
    raveled = load_raveled(raveled_path)

    target = pin_path_to_target(raveled)

    _, ax = plt.subplots(2, 2)
    target_theta = target.astype(np.float)*2*np.pi/_N_PINS
    theta = raveled.astype(np.float)*2*np.pi/_N_PINS

    ax[0][0].plot(np.sin(theta), 1-np.cos(theta), 'k-', lw=0.02)
    p1 = ax[0][1].imshow(target, aspect='auto', cmap=plt.cm.magma, interpolation='nearest')
    ax[1][0].imshow(np.sin(target_theta), aspect='auto', cmap=plt.cm.seismic, interpolation='nearest', vmin=-1, vmax=1)
    p2 = ax[1][1].imshow(np.cos(target_theta), aspect='auto', cmap=plt.cm.seismic, interpolation='nearest', vmin=-1, vmax=1)

    plt.colorbar(p1, ax=ax[0][1], label='Connection')
    plt.colorbar(p2, ax=ax[1][1], label='Position')
    ax[0][0].set_axis_off()
    ax[0][1].xaxis.set_visible(False)
    ax[1][0].xaxis.set_visible(False)
    ax[1][1].xaxis.set_visible(False)
    ax[0][1].yaxis.set_visible(False)
    ax[1][1].yaxis.set_visible(False)
    ax[0][0].set_aspect(1)
    ax[1][0].set_ylabel('Pin number')
    ax[0][1].set_title('Connections')
    ax[1][0].set_title('X coordinate')
    ax[1][1].set_title('Y coordinate')
    plt.show()

def files_to_example(base_path:str, res:int=-1) -> tf.train.Example:
    raveled = load_raveled(f"{base_path}.raveled")
    image_data = tf.io.gfile.GFile(f"{base_path}.jpg", 'rb').read()

    image = tf.io.decode_jpeg(image_data)

    if res != -1 and (image.shape[0] != res or image.shape[1] != res):
        image = tf.image.resize(image, (res, res), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image_data = tf.io.encode_jpeg(image).numpy()
        image_res = res

    assert image.shape[0] == image.shape[1], f'Invalid image shape: {image.shape}'
    image_res = image.shape[0]

    target = pin_path_to_target(raveled)
    target = tf.convert_to_tensor(target, tf.uint8)
    target = tf.reshape(target, (_N_PINS, _N_CONS, 1))
    enc_target = tf.io.encode_jpeg(target, quality=75).numpy()

    return tf.train.Example(features=tf.train.Features(feature={
        'record/version': _int_to_feature(_RECORD_FORMAT_VERSION),
        'record/name': _bytes_to_feature(base_path.encode('utf-8')),
        'image/encoded': _bytes_to_feature(image_data),
        'image/res': _int_to_feature(image_res),
        'target/encoded': _bytes_to_feature(enc_target),
        'target/n_pins': _int_to_feature(_N_PINS),
        'target/n_cons': _int_to_feature(_N_CONS),
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
                example = files_to_example(stem, self.res)
                if example is not None:
                    tfrecord_writer.write(example.SerializeToString())
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
        help='Process tfrecords one at a time (defaults to each tfrecord file in parallel process)')
    parser.add_argument('--resolution', '-r', type=int, default=-1,
        help='Resize image to this many pixels per side. Defaults to no resizing')
    parser.add_argument('input', help='Root directory for dataset (e.g. ./train)')
    parser.add_argument('output', help='Directory in which to save tfrecord files. Use "@vis" to just display images.')
    args = parser.parse_args()

    if args.output == "@vis":
        targets = list(Path(args.input).rglob('*.raveled'))
        random.shuffle(targets)
        for raveled_path in targets:
            visualize(raveled_path)

    write_records(args.input, args.output, args.num_shards, args.sequential, args.resolution)