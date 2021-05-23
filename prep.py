import glob, os
import numpy as np
import logging
import subprocess
from multiprocessing import Pool

import tensorflow as tf

import utils

def _int_to_feature(value: int) -> tf.train.Feature:
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)

def _tensor_to_feature(tensor: tf.Tensor) -> tf.train.Feature:
    serialized = tf.io.serialize_tensor(tensor)
    bytes_list = tf.train.BytesList(value=[serialized.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def d_pins(a:np.ndarray, b:np.ndarray, n_pins:int):
    a = np.array([a]).flatten().astype(np.float32) / n_pins
    b = np.array([b]).flatten().astype(np.float32) / n_pins
    dy = ((b-a)%1.0 + 0.5)%1.0 - 0.5
    return dy

def compressed_mask_pdf(path, n_pins:int, n_cons:int):
    path = path.astype(np.int)
    res = [set() for _ in range(n_pins)]

    for i in range(len(path)-1):
        a, b = path[i:i+2]

        if len(res[a]) < n_cons:
            res[a].add(b)

        if len(res[b]) < n_cons:
            res[b].add(a)

    for i in range(len(res)):
        res[i] = list(res[i])
        while len(res[i]) < n_cons:
            res[i].append(i+1)

    theta = 2 * np.pi * np.array(res) / n_pins
    res = np.array([np.sin(theta), np.cos(theta)])
    # print(res.shape)
    return res

def load_example(img_fname:str, res:int, path_len:int, n_pins:int, n_cons:int) -> tf.train.Example:
    pixels = utils.load_img(img_fname, res)

    args = ['raveler', img_fname, '-r', res, '-n', path_len, '-k', n_pins, '-w', '100e-6', '-f', 'tsv']
    sp = subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        path = np.loadtxt(sp.stdout).T[0].astype(np.int)
    except Exception as _:
        logging.warn(">> ERROR:"+ sp.stderr.read().decode('utf-8'))
        return None

    compressed_mask = compressed_mask_pdf(path, n_pins, n_cons)
    t_comp_mask = tf.convert_to_tensor(compressed_mask, dtype=tf.float32)
    t_pixels = tf.convert_to_tensor(pixels.reshape((res,res,1)), dtype=tf.float32)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/pixels': _tensor_to_feature(t_pixels),
        'image/size': _int_to_feature(res),
        'target/length': _int_to_feature(path_len),
        'target/cons': _int_to_feature(n_cons),
        'target/npins': _int_to_feature(n_pins),
        'target/compressed': _tensor_to_feature(t_comp_mask),
    }))

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

