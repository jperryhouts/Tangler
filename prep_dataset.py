import glob, os
from re import M
import numpy as np
from typing import Iterable
import subprocess
from multiprocessing import Pool
from numpy.core.fromnumeric import compress
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
    dy_norm = (np.abs(a-b)/n_pins+0.5)%1.0 - 0.5
    return (n_pins*dy_norm).astype(np.int)

def compressed_mask_pdf(a, b, n_pins:int=300, n_cons:int=10):
    res = [[]]*n_pins
    for i,j in zip(a,b):
        res[i].append(j)
        res[j].append(i)

    for i in range(len(res)):
        row = res[i][:n_cons]
        res[i] = np.pad(row, [[0,n_cons-len(row)]], constant_values=i)

    return np.array(res).astype(np.int)

def mask_pdf(a:tf.Tensor, b:tf.Tensor, n_pins:int):
    x0 = a + b*n_pins
    x1 = b + a*n_pins
    x2 = tf.concat([x0, x1], axis=-1)
    return tf.math.bincount(x2, binary_output=True, axis=-1, minlength=n_pins**2, dtype=tf.float32)

def load_example(img_fname:str, res:int, path_len:int, n_pins:int, n_cons:int) -> tf.train.Example:
    pixels = utils.load_img(img_fname, res)

    args = ['raveler', img_fname, '-r', res, '-n', path_len, '-k', n_pins, '-w', '72e-6', '-f', 'tsv']
    sp = subprocess.Popen(list(map(str, args)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        path = np.loadtxt(sp.stdout).T[0].astype(np.int)
    except Exception as e:
        print(">> ERROR:", sp.stderr.read())
        return None

    compressed_mask = compressed_mask_pdf(path[:-1], path[1:], n_pins, n_cons=n_cons)
    t_comp_mask = tf.convert_to_tensor(compressed_mask.flatten(), dtype=tf.float32)
    t_pixels = tf.convert_to_tensor(pixels.reshape((res,res,1)), dtype=tf.float32)
    t_path = tf.convert_to_tensor(path, dtype=tf.float32)
    t_mask = mask_pdf(t_path[:-1], t_path[1:], n_pins)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/pixels': _tensor_to_feature(t_pixels),
        'image/size': _int_to_feature(res),
        'target/path': _tensor_to_feature(t_path),
        'target/length': _int_to_feature(path_len),
        'target/mask': _tensor_to_feature(t_mask),
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

    n_cons = 10
    output_basename = f"res={res}_N={path_len}_k={n_pins}_nc={n_cons}_{(shard_id+1):05d}-of-{num_shards:05d}.tfrecord"
    output_path = os.path.join(output_dir, output_basename)
    print("Creating tfrecord:",output_path)

    with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
        for img_path in paths:
            example = load_example(img_path, res=res, path_len=path_len, n_pins=n_pins, n_cons=n_cons)
            if example:
                tfrecord_writer.write(example.SerializeToString())

def create_tf_records(input_dir:str, output_dir:str, num_shards:int, res:int,
                        path_len:int, n_pins:int, n_procs:int) -> None:

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
            num_shards=num_shards
        )
        shard_data.append(data)

    with Pool(n_procs) as pool:
        pool.map(create_shard, shard_data)

