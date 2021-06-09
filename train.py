import datetime, os
from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob

from model import TangledModel
import utils

def parse_example(serialized:tf.Tensor) -> dict:
    return tf.io.parse_single_example(serialized, features={
        'record/version': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=b'jpeg'),
        'image/res': tf.io.FixedLenFeature([], tf.int64),
        'target/n_pins': tf.io.FixedLenFeature([], tf.int64, default_value=256),
        'target/indices': tf.io.FixedLenFeature([], tf.string),
        'target/sparsity': tf.io.FixedLenFeature([], tf.int64),
    })

def get_decoder(res:int, n_pins:int=256, rotate:str='none'):
    assert rotate in ['none', '90', 'any']

    X,Y = np.mgrid[-1:1:res*1j,-1:1:res*1j]
    R2 = X**2+Y**2
    C_MASK = 1*(R2<1.0)
    C_MASK = C_MASK.reshape((res,res,1)).astype(np.uint8)
    C_MASK = tf.convert_to_tensor(C_MASK, dtype=tf.uint8)
    C_MASK_INV = tf.convert_to_tensor(1-C_MASK, dtype=tf.uint8)

    @tf.function()
    def decode_example(serialized:tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor]:
        example = parse_example(serialized)

        img = tf.image.decode_jpeg(example['image/encoded'])
        img = tf.reshape(img, (example['image/res'], example['image/res'], 1))
        img = C_MASK*img + 127*C_MASK_INV

        sparsity = example['target/sparsity']
        target_indices = tf.io.parse_tensor(example['target/indices'], tf.uint8)
        target_indices = tf.cast(target_indices, tf.int64)
        target_indices = tf.reshape(target_indices, (sparsity, 2))
        target_sp = tf.sparse.SparseTensor(target_indices, tf.ones(sparsity, tf.int64), (n_pins, n_pins))
        target = tf.sparse.to_dense(target_sp)
        target += tf.transpose(target)
        target = tf.clip_by_value(target, 0, 1)

        ## Rotate example
        if rotate == 'any':
            rotate_pins = tf.random.uniform([], 0, n_pins, dtype=tf.int32)
            rotate_theta = 2 * np.pi * tf.cast(rotate_pins, tf.float32) / n_pins
            img = tfa.image.rotate(img, rotate_theta, fill_mode='constant', fill_value=127)
            target = tf.roll(target, shift=rotate_pins, axis=0)
            target = tf.roll(target, shift=rotate_pins, axis=1)
        elif rotate == '90':
            rotate_k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            img = tf.image.rot90(img, k=rotate_k)
            target = tf.roll(target, shift=rotate_k*n_pins//4, axis=0)
            target = tf.roll(target, shift=rotate_k*n_pins//4, axis=1)

        return (img, target)

    return decode_example


def do_train(train_data:str, val_data:str, output_dir:str, model_name:str=None,
            checkpoint_path:str='/tmp/latest', checkpoint_period:int=1, save_format:str='h5',
            loss_function:str='mse', optimizer:str='adam', learning_rate:float=1e-3, batch_size:int=100,
            epochs:int=1, patience:int=64, train_steps:int=None, val_steps:int=None,
            use_mixed_precision:bool=False, data_cache:bool=False, vis_model:bool=False,
            dry_run:bool=False, debug:bool=False, peek:bool=False) -> None:
    tf.random.set_seed(42)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ## Load data
    if len(glob.glob(os.path.join(train_data, '*.tfrecord'))) > 0:
        FROM_TFRECORDS = True
        train_records = glob.glob(os.path.join(train_data, '*.tfrecord'))
        val_records = glob.glob(os.path.join(val_data, '*.tfrecord'))

        if debug:
            train_records = train_records[:1]
            val_records = val_records[-1:]

        ds_train = tf.data.TFRecordDataset(train_records)
        ds_val = tf.data.TFRecordDataset(val_records)

        ex = ds_train.take(1).map(parse_example).as_numpy_iterator().next()
        record_version = ex['record/version']
        assert record_version == 6, f"Incorrect record version: {record_version} != 6"
        res, n_pins = ex['image/res'], ex['target/n_pins']
    else:
        FROM_TFRECORDS = False
        res, n_pins = 256, 256

        ds_train = tf.data.Dataset.from_generator(utils.example_generator,
            args=(train_data, n_pins, res),
            output_signature=(
                tf.TensorSpec(shape=(res, res, 1), dtype=tf.uint8),
                tf.TensorSpec(shape=(n_pins,n_pins), dtype=tf.uint8),
            ))
        ds_val = tf.data.Dataset.from_generator(utils.example_generator,
            args=(val_data, n_pins, res),
            output_signature=(
                tf.TensorSpec(shape=(res,res,1), dtype=tf.uint8),
                tf.TensorSpec(shape=(n_pins,n_pins), dtype=tf.uint8),
            ))

    if not (debug or peek):
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_train = ds_train.shuffle(5*batch_size, seed=42)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    if data_cache:
        ds_train = ds_train.cache()
    if train_steps is not None:
        ds_train = ds_train.repeat()
    if data_cache:
        ds_val = ds_val.cache()
    if val_steps is not None:
        ds_val = ds_val.repeat()

    if FROM_TFRECORDS:
        ds_train = ds_train.map(get_decoder(res, n_pins, rotate='any'),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)
        ds_val = ds_val.map(get_decoder(res, n_pins, rotate='none'),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

    if peek:
        for img, target in ds_train.as_numpy_iterator():
            img = img.reshape((res,res))
            target = target.reshape((n_pins, n_pins))
            utils.plot_example(img, target)
        return

    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)

    if model_name is None:
        model_name = f"b{batch_size}_{loss_function}_{optimizer}_lr{learning_rate:g}"

    if use_mixed_precision:
        model_name += '_f16'

    model = TangledModel(res, n_pins, model_name)

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

    if loss_function == 'binary_crossentropy':
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(optimizer=opt, loss=loss_function,
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.),
                 tf.keras.metrics.AUC(from_logits=True)])

    ## Train model
    timestamp = datetime.datetime.now().strftime(r'%Y%m%d-%H%M%S')

    save_path = os.path.join(output_dir, 'models', f'{timestamp}_{model_name}')
    if vis_model:
        tf.keras.utils.plot_model(model, to_file=save_path+'.png', show_shapes=True)
    print(model.summary())
    if dry_run:
        return

    log_dir = os.path.join(output_dir, 'logs', f'{timestamp}_{model_name}')
    checkpoint_freq = 'epoch' if train_steps is None else train_steps*checkpoint_period
    checkpoint_path = f"{checkpoint_path}.{save_format}"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
            save_freq=checkpoint_freq, save_weights_only=True, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=500)]

    model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, batch_size=batch_size, epochs=epochs,
        steps_per_epoch=train_steps, validation_steps=val_steps)

    # Save model
    model.save(f"{save_path}.{save_format}", include_optimizer=False)
