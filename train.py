import datetime, os
from typing import Iterable, Tuple
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

X,Y = np.mgrid[-1:1:300j,-1:1:300j]
R2 = X**2+Y**2
C_MASK = 1*(R2<1.0)
C_MASK = C_MASK.reshape((300,300,1)).astype(np.uint8)
C_MASK = tf.convert_to_tensor(C_MASK, dtype=tf.uint8)
C_MASK_INV = tf.convert_to_tensor(1-C_MASK, dtype=tf.uint8)

def parse_example(serialized:tf.Tensor) -> dict:
    return tf.io.parse_single_example(serialized, features={
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=b'jpeg'),
        'image/res': tf.io.FixedLenFeature([], tf.int64, default_value=300),
        'target': tf.io.FixedLenFeature([], tf.string),
        'target/n_pins': tf.io.FixedLenFeature([], tf.int64, default_value=256),
        'target/n_cons': tf.io.FixedLenFeature([], tf.int64, default_value=30),
    })

@tf.function()
def decode_example(serialized:tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor]:
    example = parse_example(serialized)

    res = example['image/res']
    n_cons = example['target/n_cons']
    #n_pins = example['target/n_pins']
    #n_pins_f = tf.cast(n_pins, tf.float32)
    n_pins = 256
    n_pins_f = float(n_pins)

    img = tf.image.decode_jpeg(example['image/encoded'])
    img = tf.reshape(img, (res, res, 1))
    img *= C_MASK
    img += 127*C_MASK_INV

    target = tf.io.parse_tensor(example['target'], tf.uint8)
    target = tf.cast(target, tf.float32)
    target = tf.reshape(target, (n_pins, n_cons))

    # # # rotate_pins = tf.random.uniform([], 0, n_pins, dtype=tf.int64)
    # # # rotate_pins_f = tf.cast(rotate_pins, tf.float32)

    ## Rotate image
    # rotate_pins = random.randint(0, n_pins-1)
    # rotate_pins_f = float(rotate_pins)
    # rotate_theta = 2 * math.pi * rotate_pins_f / n_pins_f
    # img = tfa.image.rotate(img, rotate_theta, fill_mode='constant', fill_value=127)
    # target = tf.roll(target, shift=rotate_pins, axis=0)

    rotate_k = random.randint(0, 3)
    img = tf.image.rot90(img, k=rotate_k)
    target = tf.roll(target, shift=rotate_k*n_pins//4, axis=0)

    # Shift target values back to their true pin values
    ranger = tf.range(0, n_pins, 1, dtype=tf.float32)
    target += tf.reshape(ranger, (n_pins,1))

    target *= 2*math.pi/n_pins_f
    target = tf.stack([tf.math.sin(target), tf.math.cos(target)], axis=0)

    return (img, target)

def get_data_shape(ds):
    def get_length(serialized:tf.Tensor) -> Tuple[int,int]:
        example = parse_example(serialized)
        return (example['image/res'], example['target/n_pins'], example['target/n_cons'])
    return ds.take(1).map(get_length).as_numpy_iterator().next()

def do_train(train_records:Iterable[str], val_records:Iterable[str], output_dir:str,
            model_name:str=None, checkpoint_path:str='/tmp/latest', checkpoint_period:int=1,
            loss_function:str='mse', optimizer:str='adam', learning_rate:float=1e-3,
            batch_size:int=100, use_mixed_precision:bool=False,
            epochs:int=1, patience:int=30, train_steps:int=None, val_steps:int=None,
            data_cache:bool=False, vis_model:bool=False, save_format:str='h5') -> None:
    tf.random.set_seed(42)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ## Load data
    raw_dataset_train = tf.data.TFRecordDataset(train_records, num_parallel_reads=8)
    res, n_pins, n_cons = get_data_shape(raw_dataset_train)

    ds_train = raw_dataset_train.prefetch(tf.data.AUTOTUNE)
    ## Cache before decoding example, so that rotation is different on each epoch
    if data_cache:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(10*batch_size, seed=42)
    ds_train = ds_train.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size)

    raw_dataset_val = tf.data.TFRecordDataset(val_records, num_parallel_reads=8)
    ds_val = raw_dataset_val.prefetch(tf.data.AUTOTUNE)
    if data_cache:
        ds_val = ds_val.cache()
    ds_val = ds_val.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)

    ## Define model
    preprocess_layers = [
        tf.keras.layers.experimental.preprocessing.Rescaling(-2*math.pi/255.0, offset=math.pi, name='scale_invert'),
    ]

    grouped_convolutional_layers = [
        # [
        #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #     tf.keras.layers.Flatten(name='bypass'),
        # ],
        [
            tf.keras.layers.Conv2D(filters=10, kernel_size=6, padding='VALID', strides=(3,3)),
            tf.keras.layers.Flatten(name='conv_f10.k6.s3'),
        ],
    ]

    hidden_layers = [
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(5000, activation='relu'),
        tf.keras.layers.Dense(n_pins*n_cons, activation='linear'),
        tf.keras.layers.Reshape((1, n_pins, n_cons)),
    ]

    grouped_postprocess_layers = [
        [ tf.keras.layers.Activation(tf.math.sin, name='x_pos') ],
        [ tf.keras.layers.Activation(tf.math.cos, name='y_pos') ],
    ]

    ## Assemble model
    if model_name is None:
        model_name = f"b{batch_size}_{loss_function}_{optimizer}_lr{learning_rate:g}"

    if use_mixed_precision:
        model_name += '_f16'

    sequence = [tf.keras.Input(shape=(res,res,1), dtype='uint8')]

    for layer in preprocess_layers:
        sequence.append(layer(sequence[-1]))

    convolutional_layers = []
    for group in grouped_convolutional_layers:
        sub_sequence = [group[0](sequence[-1])]
        for layer in group[1:]:
            sub_sequence.append(layer(sub_sequence[-1]))
        convolutional_layers.append(sub_sequence[-1])

    if len(convolutional_layers) > 1:
        sequence.append(tf.keras.layers.Concatenate(axis=-1)(convolutional_layers))
    else:
        sequence.append(convolutional_layers[0])

    for layer in hidden_layers:
        sequence.append(layer(sequence[-1]))

    postprocess_layers = []
    for group in grouped_postprocess_layers:
        sub_sequence = [group[0](sequence[-1])]
        for layer in group[1:]:
            sub_sequence.append(layer(sub_sequence[-1]))
        postprocess_layers.append(sub_sequence[-1])

    if len(postprocess_layers) > 1:
        sequence.append(tf.keras.layers.Concatenate(axis=1)(postprocess_layers))
    else:
        sequence.append(postprocess_layers[0])

    model = tf.keras.models.Model(inputs=sequence[0], outputs=sequence[-1], name=model_name)

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer == 'adam_amsgrad':
        opt = tf.keras.optimizers.Adam(learning_rate, amsgrad=True)
    elif optimizer == 'nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate)
    elif optimizer == 'adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss=loss_function)

    ## Train model
    timestamp = datetime.datetime.now().strftime(r'%Y%m%d-%H%M%S')

    save_path = os.path.join(output_dir, 'models', f'{timestamp}_{model_name}')
    if vis_model:
        tf.keras.utils.plot_model(model, to_file=save_path+'.png', show_shapes=True)
    print(model.summary())

    log_dir = os.path.join(output_dir, 'logs', f'{timestamp}_{model_name}')
    checkpoint_freq = 'epoch' if train_steps is None else train_steps*checkpoint_period
    checkpoint_path = f"{checkpoint_path}.{save_format}"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_freq=checkpoint_freq),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=500)]

    model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, batch_size=batch_size, epochs=epochs,
        steps_per_epoch=train_steps, validation_steps=val_steps)

    # Save model
    model.save(f"{save_path}.{save_format}", include_optimizer=False)
