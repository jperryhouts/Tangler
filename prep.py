import glob, os
from typing import Iterable, Mapping, Any
import numpy as np
import logging
import subprocess
from multiprocessing import Pool

import tensorflow as tf

import utils

# def ravel(img_path:str, res:int, pattern_len:int, n_pins:int, weight:float=100e-6) -> np.ndarray:
#     args = ['raveler', img_path, '-r', res, '-n', pattern_len, '-k', n_pins, '-w', weight, '-f', 'tsv']
#     sp = subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     try:
#         pattern = np.loadtxt(sp.stdout).T[0].astype(np.int)
#     except Exception as _:
#         logging.warn(">> ERROR:"+ sp.stderr.read().decode('utf-8'))
#         return None
#     return pattern

def load_example(img_path:str, res:int, n_pins:int, n_cons:int) -> tf.train.Example:
    raveled_fname = f"{img_path[:-5]}.raveled"
    pattern = np.loadtxt(raveled_fname, dtype=np.int).flatten()
    if not pattern or len(pattern) == 0:
        logging.error(f">> Unable to load raveled sequence <{raveled_fname}>")
        return None
    if pattern.min() < 0 or pattern.max() >= n_pins:
        logging.error(f">> Value out of bounds in raveled sequence <{raveled_fname}>")
        return None

    # pattern = ravel(img_path, res, pattern_len, n_pins)
    # if pattern is None:
    #     return None

    target = utils.pin_path_to_target(pattern, n_pins, n_cons)

    image = utils.load_img(img_path, res).astype(np.uint8)
    image = image.reshape((res,res,1))

    return utils.encode_example(image, target)

def create_shard(args:Mapping[str:Any]):
    shard_id = args['shard_id']
    paths = args['paths']
    res = args['res']
    n_pins = args['n_pins']
    pattern_len = args['pattern_len']
    num_shards = args['num_shards']
    output_dir = args['output_dir']
    n_cons = args['n_cons']

    output_basename = f"r{res}_N{pattern_len}_k{n_pins}_c{n_cons}_{(shard_id+1):05d}-of-{num_shards:05d}.tfrecord"
    output_path = os.path.join(output_dir, output_basename)
    print(f">> Creating tfrecord: f{output_path}")

    with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
        for img_path in paths:
            example = load_example(img_path, res=res, n_pins=n_pins, n_cons=n_cons)
            if example:
                tfrecord_writer.write(example.SerializeToString())


def do_prep(input_dir:str, output_dir:str,
            res:int, pattern_len:int, n_pins:int, n_cons:int,
            num_shards:int=10, n_procs:int=1) -> None:

    assert os.path.isdir(input_dir)
    fnames = glob.glob(os.path.join(input_dir, '*', '*.JPEG'))
    assert len(fnames) > 0

    np.random.shuffle(fnames)

    if n_cons <= 0:
        n_cons = 2 * pattern_len // n_pins

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.isdir(output_dir)

    num_per_shard = int(np.ceil(len(fnames)/float(num_shards)))

    shard_data = []
    for i in range(num_shards):
        data = dict(
            shard_id = i,
            paths=fnames[i*num_per_shard:(i+1)*num_per_shard],
            res=res,
            n_pins=n_pins,
            pattern_len=pattern_len,
            output_dir=output_dir,
            num_shards=num_shards,
            n_cons=n_cons
        )
        shard_data.append(data)

    with Pool(n_procs) as pool:
        pool.map(create_shard, shard_data)

