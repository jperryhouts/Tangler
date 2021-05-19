import datetime, os
from typing import List
import tensorflow as tf
import numpy as np

import utils

class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

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

def do_train(tfrecords: List[str], res: int, path_len: int, output_dir: str,
            n_pins: int=300, batch_size: int=10, epochs: int=1,
            overshoot_epochs: int=30, random_seed: int=42) -> None:
    tf.random.set_seed(random_seed)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    ## Assemble model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(res,res,1)))
    model.add(utils.ScaleLayer(scale=1.0/127.5, offset=-127.5, normalize=False, name='normalize_pixels'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2*(path_len+1), activation='gelu'))
        #activity_regularizer=tf.keras.regularizers.L2(l2=0.001)))
    model.add(utils.ScaleLayer(scale=1.0, offset=0, normalize=True, name='normalize_cartesian'))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(utils.NormalizeCartesianRadius())
    # model.add(tf.keras.layers.Dense(path_len+1, activation='linear'))
    #model.add(tf.keras.layers.Dense(2*(path_len+1), activation='linear',
    #    activity_regularizer=tf.keras.regularizers.L2(l2=0.001)))
    # model.add(utils.RadialSigmoid())
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Dense(2*(path_len+1), activation=tf.nn.leaky_relu))
    # model.add(utils.ScaleLayer(scale=n_pins, offset=0.0, normalize=False, name='scale_shift_polar'))
    #model.add(utils.ScaleLayer(scale=1.0, offset=0, normalize=True, name='normalize_cartesian'))
    model.add(utils.CartesianToPolar())
    model.add(tf.keras.layers.Dense(path_len+1, activation='linear'))
    model.add(utils.ScaleLayer(scale=n_pins, normalize=True, name='normalize_polar'))
    # model.add(tf.keras.layers.Dense(path_len+1, activation='relu'))
    # model.add(utils.ScaleLayer(scale=n_pins, normalize=False, name='scale_shift_polar'))

    # model.add(tf.keras.layers.Dense(path_len, activation='relu',
    #     kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)
    #     ))

    #model.add(tf.keras.layers.SpatialDropout2D(0.5))
    #model.add(tf.keras.layers.Dense(res*res*0.25, activation='relu'))
    #model.add(tf.keras.layers.Dense(1000, activation='relu'))
    ##model.add(tf.keras.layers.Dense(1000, activation='relu'))
    #model.add(tf.keras.layers.Dense(1000, activation='relu'))
    #model.add(tf.keras.layers.Dense(1000, activation='relu',
    #    kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)
    #    ))
    #model.add(utils.ScaleLayer(scale=3, offset=0, normalize=False, name='scale_shift_layer'))
    #model.add(tf.keras.layers.Dense(1000, activation='linear',
        #kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)
    #    ))
    #model.add(tf.keras.layers.Dense(path_len, activation=tf.math.sin))
    #model.add(tf.keras.layers.Lambda(utils.lambda_activation, output_shape=(path_len,)))
    #model.add(utils.ScaleLayer(scale=10*n_pins, offset=0, normalize=False, name='scale_shift_layer'))


    ## Load dataset
    raw_dataset_train = tf.data.TFRecordDataset(tfrecords)
    parser = get_proto_parser(path_len)
    dataset_train = raw_dataset_train.map(parser).shuffle(batch_size*5)
    batched_train = dataset_train.batch(batch_size, drop_remainder=True)

    ## Train model
    masks = utils.Masks(n_pins)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss=utils.index_error)
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=masks.loss_function)

    logdir = os.path.join(output_dir, 'logs', timestamp)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=overshoot_epochs),
                tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'models', timestamp+'.latest.tf'),
                                                    monitor='loss', save_best_only=False),
                tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_dir, 'models', timestamp+'.best.tf'),
                                                    monitor='loss', save_best_only=True),
                LRTensorBoard(log_dir=logdir)]

    model.fit(batched_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
