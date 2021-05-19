import datetime, os
from typing import Iterable
import tensorflow as tf

import utils

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

def do_train(train_records:Iterable[str], res:int, path_len:int, output_dir:str,
            name:str=None, n_pins:int=300, batch_size:int=10, epochs:int=1,
            overshoot_epochs:int=30, random_seed:int=42) -> None:
    tf.random.set_seed(random_seed)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    ## Assemble model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(res,res,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(path_len+1, activation='relu', name='dense_relu_01'))
    model.add(tf.keras.layers.Dense(path_len+1, activation='relu', name='dense_relu_02'))
    model.add(tf.keras.layers.Dense(path_len+1, activation='relu', name='dense_relu_03'))
    model.add(tf.keras.layers.Dense(path_len+1, activation='relu', name='dense_relu_04'))
    model.add(tf.keras.layers.Dense(path_len+1, activation='relu', name='dense_relu_05'))
    model.add(tf.keras.layers.Dense(path_len+1, activation=tf.math.sin, name='dense_sine'))
    model.add(tf.keras.layers.Dense(path_len+1, activation='linear', name='dense_linear'))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(n_pins))

    ## Load dataset
    parser = get_proto_parser(path_len)

    raw_dataset_train = tf.data.TFRecordDataset(train_records)
    dataset_train = raw_dataset_train.map(parser).shuffle(batch_size*5)
    batched_train = dataset_train.batch(batch_size, drop_remainder=True)

    ## Train model
    loss_function = utils.get_loss_function(n_pins)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss_function)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=overshoot_epochs),
                #tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/latest.tf', monitor='loss', save_best_only=True),
                tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs', f'{timestamp}_{name}'))]

    model.fit(batched_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    model.save(os.path.join(output_dir, 'models', f'{timestamp}_{name}.final.tf'))
