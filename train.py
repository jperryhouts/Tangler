import datetime, os
from typing import Iterable, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

def parse_example(serialized:tf.Tensor) -> dict:
    return tf.io.parse_single_example(serialized, features={
        'record/version': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=b'jpeg'),
        'image/res': tf.io.FixedLenFeature([], tf.int64),
        'target/encoded': tf.io.FixedLenFeature([], tf.string),
        'target/n_pins': tf.io.FixedLenFeature([], tf.int64, default_value=256),
        'target/n_cons': tf.io.FixedLenFeature([], tf.int64, default_value=64),
    })

def get_decoder(res:int, n_pins:int=256, n_cons:int=64, rotate:str='none'):
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

        target_data = tf.image.decode_jpeg(example['target/encoded'])
        target = tf.cast(target_data, tf.float32)
        target = tf.reshape(target, (n_pins, n_cons))

        ## Rotate example
        if rotate == 'any':
            rotate_pins = tf.random.uniform([], 0, n_pins, dtype=tf.int32)
            rotate_theta = 2 * np.pi * tf.cast(rotate_pins, tf.float32) / n_pins
            img = tfa.image.rotate(img, rotate_theta, fill_mode='constant', fill_value=127)
            target = tf.roll(target, shift=rotate_pins, axis=0)
        elif rotate == '90':
            rotate_k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            img = tf.image.rot90(img, k=rotate_k)
            target = tf.roll(target, shift=rotate_k*n_pins//4, axis=0)

        ## Shift target values back to their true pin values
        # ranger = tf.range(0, n_pins, 1, dtype=tf.float32)
        # target += tf.reshape(ranger, (n_pins,1))

        ## Evaluate cartesian coordinates of target pins
        target /= float(n_pins)
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
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Autoencoder(inputs):
    down_stack = [
        #downsample(64, 4), #, apply_batchnorm=False),  # (bs, 128, 128, 64)
        #downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4, apply_batchnorm=False),  # (bs, 32, 32, 256)
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
        #upsample(128, 4),  # (bs, 64, 64, 256)
        #upsample(64, 4),  # (bs, 128, 128, 128)
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

    a = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same')(x)
    b = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same')(x)
    c = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same')(x)
    d = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same')(x)

    outputs = tf.keras.layers.Concatenate(axis=1)([a,b,c,d])

    return outputs #tf.keras.Model(inputs=inputs, outputs=x)

def get_data_shape(ds):
    def get_length(serialized:tf.Tensor) -> Tuple[int,int]:
        example = parse_example(serialized)
        return (example['image/res'], example['target/n_pins'], example['target/n_cons'], example['record/version'])
        # return (256, example['target/n_pins'], example['target/n_cons'])
    return ds.take(1).map(get_length).as_numpy_iterator().next()

def do_train(train_records:Iterable[str], val_records:Iterable[str], output_dir:str,
            model_name:str=None, checkpoint_path:str='/tmp/latest', checkpoint_period:int=1,
            loss_function:str='mse', optimizer:str='adam', learning_rate:float=1e-3,
            batch_size:int=100, use_mixed_precision:bool=False,
            epochs:int=1, patience:int=64, train_steps:int=None, val_steps:int=None,
            data_cache:bool=False, vis_model:bool=False, save_format:str='h5', dry_run:bool=False) -> None:
    tf.random.set_seed(42)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ## Load data
    raw_dataset_train = tf.data.TFRecordDataset(train_records)
    res, n_pins, n_cons, record_version = get_data_shape(raw_dataset_train)
    assert record_version == 3

    ds_train = raw_dataset_train.prefetch(tf.data.AUTOTUNE)
    ## Cache before decoding example, so that rotation is different on each epoch
    if data_cache:
        ds_train = ds_train.cache()
    if train_steps is not None:
        ds_train = ds_train.repeat()
    ds_train = ds_train.shuffle(5*batch_size, seed=17)
    ds_train = ds_train.map(get_decoder(res, n_pins, n_cons, rotate='none'), num_parallel_calls=tf.data.AUTOTUNE)

    # import sys, matplotlib.pyplot as plt
    # for img, target in ds_train.take(3):
    #     _, ax = plt.subplots(1, 3)
    #     ax[0].imshow(img.numpy().reshape((res,res)), aspect=1, cmap=plt.cm.gray, vmin=0, vmax=255)
    #     ax[1].imshow(target.numpy()[0].reshape((n_pins,n_cons)), aspect=n_cons/n_pins, cmap=plt.cm.coolwarm, interpolation='nearest', vmin=-1, vmax=1)
    #     ax[2].imshow(target.numpy()[1].reshape((n_pins,n_cons)), aspect=n_cons/n_pins, cmap=plt.cm.coolwarm, interpolation='nearest', vmin=-1, vmax=1)
    #     plt.show()
    # if np.zeros(1)[0] == 0:
    #     sys.exit()

    ds_train = ds_train.batch(batch_size)

    raw_dataset_val = tf.data.TFRecordDataset(val_records)
    ds_val = raw_dataset_val.prefetch(tf.data.AUTOTUNE)
    if data_cache:
        ds_val = ds_val.cache()
    if val_steps is not None:
        ds_val = ds_val.repeat()
    ds_val = ds_val.map(get_decoder(res, n_pins, n_cons, rotate='none'), num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)

    ## Define model
    preprocess_layers = [
        tf.keras.layers.experimental.preprocessing.Rescaling(-1./127.5, offset=1.0),

        tf.keras.layers.Conv2D(64, 4, strides=1, padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(128, 4, strides=1, padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # tf.keras.layers.LeakyReLU(),
        #tf.keras.layers.Conv2D(128, int(1+res//4-(2**int(np.log2(res//4)))), padding='valid'),
    ]

    grouped_convolutional_layers = [
        [
            Autoencoder,
        ],
        # [
        #     tf.keras.layers.Conv2DTranspose(32, 4, strides=1, padding='same'),
        # ]
    ]

    hidden_layers = [
        # tf.keras.layers.Flatten(),
        # # tf.keras.layers.Dropout(0.5),
        # # tf.keras.layers.Dense(2*n_pins, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # tf.keras.layers.LeakyReLU(),
        # tf.keras.layers.Dense(n_pins*n_cons),

        #tf.keras.layers.Reshape((1, n_pins, n_cons)),
        tf.keras.layers.Reshape((n_pins, n_cons)),
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

    model.compile(optimizer=opt, loss=loss_function, metrics=['mse', 'mae'])

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
