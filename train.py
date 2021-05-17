#!/usr/bin/env python3

import glob, os
from argparse import ArgumentParser


import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()

from utils import Masks

IMG_RES = 150
PATH_LEN = 1000
N_PINS = 300

def parse_proto(proto):
    schema = {
        'image/pixels': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/size': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'target': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    example = tf.io.parse_single_example(proto, schema)

    size = example['image/size']

    pixels = tf.io.parse_tensor(example['image/pixels'], tf.float32)
    pixels = tf.reshape(pixels, (size, size, 1))

    target = tf.io.parse_tensor(example['target'], tf.float32)[:PATH_LEN]

    return [pixels, target]


def train(datadir):
    masks = Masks(N_PINS)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_RES,IMG_RES,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(PATH_LEN, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(PATH_LEN, activation='linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=masks.loss_function)

    batch_size = 10

    train_fnames = glob.glob(os.path.join(datadir, 'train', '*.tfrecord'))
    raw_dataset_train = tf.data.TFRecordDataset(train_fnames)
    dataset_train = raw_dataset_train.map(parse_proto).shuffle(batch_size*5)
    batched_train = dataset_train.batch(batch_size, drop_remainder=True)

    model.fit(batched_train, epochs=1, batch_size=batch_size)
    return model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', '-o', help='Save model to path')
    parser.add_argument('data', help='path to tfrecord dataset directory')
    args = parser.parse_args()

    model = train(args.data)
    if (args.output):
        model.save(args.output)

