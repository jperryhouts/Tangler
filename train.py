from typing import List
import tensorflow as tf
from tensorflow.python.data.util import random_seed
from tensorflow.python.ops.variables import initialize_variables

from utils import Masks

def get_proto_parser(path_len:int):
    def parser(proto):
        example = tf.io.parse_single_example(proto, {
            'image/pixels': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/size': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'target': tf.io.FixedLenFeature([], tf.string, default_value='')
        })

        pixels = tf.io.parse_tensor(example['image/pixels'], tf.float32)
        target = tf.io.parse_tensor(example['target'], tf.float32)
        res = example['image/size']
        return [tf.reshape(pixels, (res,res,1)), target[:path_len+1]]

    return parser

def scale(factor):
    @tf.function
    def activation(inputs):
        return factor*inputs
    return activation

def do_train(tfrecords: List[str], res: int, path_len: int,
            n_pins: int=300, batch_size: int=10, epochs: int=1,
            random_seed: int=42) -> tf.keras.Model:
    tf.random.set_seed(random_seed)

    ## Assemble model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(res,res,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu',
        activity_regularizer=tf.keras.regularizers.L2(l2=0.01),
        kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)))
    model.add(tf.keras.layers.Dense(path_len, activation='linear'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(scale(n_pins)))

    ## Load dataset
    raw_dataset_train = tf.data.TFRecordDataset(tfrecords)
    parser = get_proto_parser(path_len)
    dataset_train = raw_dataset_train.map(parser).shuffle(batch_size*5)
    batched_train = dataset_train.batch(batch_size, drop_remainder=True)

    ## Train model
    masks = Masks(n_pins)
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=masks.loss_function)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=masks.loss_function)
    model.fit(batched_train, epochs=epochs, batch_size=batch_size)
    return model
