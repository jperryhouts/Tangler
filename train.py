import datetime, os
from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob

# from utils import periodic_padding
from utils import example_generator

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

        #target = tf.image.decode_jpeg(example['target/encoded'])
        sparsity = example['target/sparsity']
        target_indices = tf.io.parse_tensor(example['target/indices'], tf.uint8)
        target_indices = tf.cast(target_indices, tf.int64)
        target_indices = tf.reshape(target_indices, (sparsity, 2))
        target_sp = tf.sparse.SparseTensor(target_indices, tf.ones(sparsity, tf.int64), (n_pins, n_pins))
        target = tf.sparse.to_dense(target_sp)
        target += tf.transpose(target)

        #target_serialized = tf.expand_dims(example['target/sparse'], axis=0)
        #target_sparse = tf.io.deserialize_many_sparse(target_serialized, dtype=tf.uint8)
        #target = tf.squeeze(tf.sparse.to_dense(target_sparse))

        # target = tf.io.parse_tensor(example['target'], tf.uint8)
        #target = tf.cast(target, tf.int64)
        #target = tf.clip_by_value(target, 0, 1)
        #target = tf.reshape(target, (n_pins, n_pins))
        # target -= tf.linalg.band_part(target, 15, 15)

        # ## Rotate example
        # if rotate == 'any':
        #     rotate_pins = tf.random.uniform([], 0, n_pins, dtype=tf.int32)
        #     rotate_theta = 2 * np.pi * tf.cast(rotate_pins, tf.float32) / n_pins
        #     img = tfa.image.rotate(img, rotate_theta, fill_mode='constant', fill_value=127)
        #     target = tf.roll(target, shift=rotate_pins, axis=0)
        # elif rotate == '90':
        #     rotate_k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        #     img = tf.image.rot90(img, k=rotate_k)
        #     target = tf.roll(target, shift=rotate_k*n_pins//4, axis=0)

        ## Shift target values back to their true pin values
        # ranger = tf.range(0, n_pins, 1, dtype=tf.float32)
        # target += tf.reshape(ranger, (n_pins,1))

        ## Evaluate cartesian coordinates of target pins
        # target /= float(n_pins)
        # target *= 2*np.pi/float(n_pins)
        # target = tf.stack([tf.math.sin(target), tf.math.cos(target)], axis=0)

        return (img, target)

    return decode_example

## The following three functions are from the TensorFlow Pix2Pix tutorial:
## https://www.tensorflow.org/tutorials/generative/pix2pix

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
        padding='same', kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.Activation(tf.keras.activations.swish))

    return result

def Autoencoder(inputs):
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    outputs = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', # activation='sigmoid',
        kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)

    return outputs

def get_data_shape(ds):
    def get_length(serialized:tf.Tensor) -> Tuple[int,int]:
        example = parse_example(serialized)
        return (example['image/res'], example['target/n_pins'], example['record/version'])
    return ds.take(1).map(get_length).as_numpy_iterator().next()

def do_train(train_data:str, val_data:str, output_dir:str, model_name:str=None,
            checkpoint_path:str='/tmp/latest', checkpoint_period:int=1, save_format:str='h5',
            loss_function:str='mse', optimizer:str='adam', learning_rate:float=1e-3, batch_size:int=100,
            epochs:int=1, patience:int=64, train_steps:int=None, val_steps:int=None,
            use_mixed_precision:bool=False, data_cache:bool=False,
            vis_model:bool=False, dry_run:bool=False, debug:bool=False) -> None:
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
        res, n_pins, record_version = get_data_shape(ds_train)
        assert record_version == 6

        ds_val = tf.data.TFRecordDataset(val_records)
    else:
        FROM_TFRECORDS = False
        res = 256
        n_pins = 256

        ds_train = tf.data.Dataset.from_generator(example_generator,
            args=(train_data, n_pins, res),
            output_signature=(
                tf.TensorSpec(shape=(res, res, 1), dtype=tf.uint8),
                tf.TensorSpec(shape=(n_pins,n_pins), dtype=tf.uint8),
            ))
        ds_val = tf.data.Dataset.from_generator(example_generator,
            args=(val_data, n_pins, res),
            output_signature=(
                tf.TensorSpec(shape=(res,res,1), dtype=tf.uint8),
                tf.TensorSpec(shape=(n_pins,n_pins), dtype=tf.uint8),
            ))

    #ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    #ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
    #ds_train = ds_train.shuffle(5*batch_size, seed=42)

    if data_cache:
        ds_train = ds_train.cache()
    if train_steps is not None:
        ds_train = ds_train.repeat()
    if data_cache:
        ds_val = ds_val.cache()
    if val_steps is not None:
        ds_val = ds_val.repeat()

    if FROM_TFRECORDS:
        tfrecord_decoder = get_decoder(res, n_pins, rotate='none')
        ds_train = ds_train.map(tfrecord_decoder,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)
        ds_val = ds_val.map(tfrecord_decoder,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

    import matplotlib.pyplot as plt
    for img, target in ds_train.as_numpy_iterator():
        _, ax = plt.subplots(1, 3, figsize=(14,4))
        img = img.astype(np.float32)
        target = target.astype(np.float32)
        ax[0].imshow(img.reshape((res,res)), aspect=1, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax[1].imshow((target).reshape((n_pins,n_pins)), aspect=1, cmap=plt.cm.gray_r, interpolation='nearest')

        target = np.tril(target)
        w = np.where(target > 0.5)
        path = []
        for i in range(w[0].size):
            path.append(w[0][i])
            path.append(w[1][i])

        print(len(path))
        raveled = np.array(path).astype(np.float)
        theta = raveled*2*np.pi/256

        ax[2].plot(np.sin(theta), 1-np.cos(theta), 'k-', lw=0.01)

        #plt.colorbar(tp, ax=ax[1])
        plt.show()

    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)

    ## Define model
    preprocess_layers = [
        tf.keras.layers.experimental.preprocessing.Rescaling(-1./127.5, offset=1.0),
    ]

    grouped_convolutional_layers = [
    ]

    hidden_layers = [
        Autoencoder,
        # tf.keras.layers.Flatten(),
        # # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Reshape((n_pins, n_pins)),
    ]

    grouped_postprocess_layers = [
        # [ tf.keras.layers.Activation(tf.math.sin, name='x_pos') ],
        # [ tf.keras.layers.Activation(tf.math.cos, name='y_pos') ],
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
    elif len(convolutional_layers) == 1:
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
    elif len(postprocess_layers) == 1:
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

    if loss_function == 'binary_crossentropy':
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(optimizer=opt, loss=loss_function,
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.)])

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
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_freq=checkpoint_freq),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=500)]

    model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, batch_size=batch_size, epochs=epochs,
        steps_per_epoch=train_steps, validation_steps=val_steps)

    # Save model
    model.save(f"{save_path}.{save_format}", include_optimizer=False)
