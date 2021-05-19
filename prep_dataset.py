import glob, os
import numpy as np
import tensorflow as tf

import utils

def _int_to_feature(value: int) -> tf.train.Feature:
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)

def _tensor_to_feature(tensor: tf.Tensor) -> tf.train.Feature:
    serialized = tf.io.serialize_tensor(tensor)
    bytes_list = tf.train.BytesList(value=[serialized.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def load_example(path_stem: str, res:int) -> tf.train.Example:
    pixels = utils.load_img(path_stem+'.JPEG', res)
    path = np.loadtxt(path_stem+'.tsv')

    t_pixels = tf.convert_to_tensor(pixels.reshape((res,res,1)), dtype=tf.float32)
    t_path = tf.convert_to_tensor(path.flatten(), dtype=tf.float32)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/pixels': _tensor_to_feature(t_pixels),
        'image/size': _int_to_feature(res),
        'target': _tensor_to_feature(t_path)
    }))

def create_tf_records(input_dir:str, output_dir:str, num_shards:int, res:int) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert os.path.isdir(input_dir)
    assert os.path.isdir(output_dir)

    fnames = glob.glob(os.path.join(input_dir, '*', '*.JPEG'))

    n_images = len(fnames)
    num_per_shard = int(np.ceil(n_images/float(num_shards)))

    for shard_id in range(num_shards):
        output_basename = f"res={res}_{(shard_id+1):05d}-of-{num_shards:05d}.tfrecord"
        output_path = os.path.join(output_dir, output_basename)

        with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min(start_idx+num_per_shard, n_images)
            for i in range(start_idx, end_idx):
                img_fname = fnames[i]
                path_stem = img_fname.rsplit('.', 1)[0]
                example = load_example(path_stem, res)
                tfrecord_writer.write(example.SerializeToString())
