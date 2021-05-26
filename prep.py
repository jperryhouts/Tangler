import glob, os
import numpy as np
import logging
import subprocess
from multiprocessing import Pool

import tensorflow as tf

import utils

def load_example(img_fname:str, res:int, path_len:int, n_pins:int, n_cons:int) -> tf.train.Example:
    args = ['raveler', img_fname, '-r', 600, '-n', path_len, '-k', n_pins, '-w', '100e-6', '-f', 'tsv']
    sp = subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        path = np.loadtxt(sp.stdout).T[0].astype(np.int)
        target = utils.pin_path_to_target(path, n_pins, n_cons)
    except Exception as _:
        logging.warn(">> ERROR:"+ sp.stderr.read().decode('utf-8'))
        return None

    image = utils.load_img(img_fname, res).astype(np.uint8)
    image = image.reshape((res,res,1))

    return utils.encode_example(image, target)

def create_shard(data):
    shard_id = data['shard_id']
    paths = data['paths']
    res = data['res']
    n_pins = data['n_pins']
    path_len = data['path_len']
    num_shards = data['num_shards']
    output_dir = data['output_dir']
    n_cons = data['n_cons']

    output_basename = f"r{res}_N{path_len}_k{n_pins}_c{n_cons}_{(shard_id+1):05d}-of-{num_shards:05d}.tfrecord"
    output_path = os.path.join(output_dir, output_basename)
    print(f">> Creating tfrecord: f{output_path}")

    with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
        for img_path in paths:
            example = load_example(img_path, res=res, path_len=path_len, n_pins=n_pins, n_cons=n_cons)
            if example:
                tfrecord_writer.write(example.SerializeToString())

def create_tf_records(input_dir:str, output_dir:str, num_shards:int, res:int,
                        path_len:int, n_pins:int, n_cons:int, n_procs:int=1) -> None:

    if n_cons <= 0:
        n_cons = 2 * path_len // n_pins

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.isdir(input_dir)
    assert os.path.isdir(output_dir)

    fnames = glob.glob(os.path.join(input_dir, '*', '*.JPEG'))
    assert len(fnames) > 0
    np.random.shuffle(fnames)

    num_per_shard = int(np.ceil(len(fnames)/float(num_shards)))

    shard_data = []
    for i in range(num_shards):
        data = dict(
            shard_id = i,
            paths=fnames[i*num_per_shard:(i+1)*num_per_shard],
            res=res,
            n_pins=n_pins,
            path_len=path_len,
            output_dir=output_dir,
            num_shards=num_shards,
            n_cons=n_cons
        )
        shard_data.append(data)

    with Pool(n_procs) as pool:
        pool.map(create_shard, shard_data)

