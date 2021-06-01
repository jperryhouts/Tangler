import datetime, os
from typing import Iterable, Tuple
import math
import tensorflow as tf

_PATH_LEN = 3000

_FEATURES = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/name': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=b'jpeg'),
    'image/res': tf.io.FixedLenFeature([], tf.int64, default_value=300),
    'target/sequence': tf.io.FixedLenFeature([], tf.string),
    'target/length': tf.io.FixedLenFeature([], tf.int64, default_value=_PATH_LEN),
}

@tf.function
def decode_example(serialized:tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor]:
    example = tf.io.parse_single_example(serialized, _FEATURES)

    res = example['image/res']
    img = tf.image.decode_jpeg(example['image/encoded'])
    img = tf.reshape(img, (res, res, 1))

    target = tf.io.parse_tensor(example['target/sequence'], tf.uint8)
    target = tf.cast(target[:_PATH_LEN], tf.float32)
    target += tf.random.normal((_PATH_LEN,), mean=0.0, stddev=0.5)
    target = 2*math.pi*target/256
    target = tf.stack([tf.math.sin(target), tf.math.cos(target)], axis=0)

    return (img, target)

def get_data_shape(ds):
    def get_length(serialized:tf.Tensor) -> Tuple[int,int]:
        example = tf.io.parse_single_example(serialized, _FEATURES)
        #return (example['image/res'], example['target/length'])
        return (example['image/res'], _PATH_LEN)
    res, length = ds.take(1).map(get_length).as_numpy_iterator().next()
    return (res, length)

def do_train(train_records:Iterable[str], val_records:Iterable[str], output_dir:str,
            model_name:str=None, checkpoint_path:str='/tmp/latest.tf', checkpoint_period:int=1,
            loss_function:str='mse', optimizer:str='adam', learning_rate:float=1e-3,
            batch_size:int=100, use_mixed_precision:bool=False,
            epochs:int=1, patience:int=30, train_steps_per_epoch:int=None, val_steps:int=None,
            data_cache:bool=False, vis_model:bool=False) -> None:
    tf.random.set_seed(42)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ## Load data
    raw_dataset_train = tf.data.TFRecordDataset(train_records, num_parallel_reads=8)
    res, path_len = get_data_shape(raw_dataset_train)

    ds_train = raw_dataset_train.prefetch(tf.data.AUTOTUNE)
    ds_train = raw_dataset_train.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    if data_cache:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(5*batch_size, seed=42).batch(batch_size)

    raw_dataset_val = tf.data.TFRecordDataset(val_records, num_parallel_reads=8)
    ds_val = raw_dataset_train.prefetch(tf.data.AUTOTUNE)
    ds_val = raw_dataset_val.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    if data_cache:
        ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size)

    ## Define model
    preprocess_layers = [
        tf.keras.layers.experimental.preprocessing.Rescaling(-2*math.pi/255, offset=math.pi, name='scale_invert'),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
    ]

    grouped_convolutional_layers = [
        [
            tf.keras.layers.Flatten(name='bypass'),
        ],
        [
            tf.keras.layers.Conv2D(30, 3, padding='valid'),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            tf.keras.layers.Flatten(name='conv_30'),
        ],
    ]

    hidden_layers = [
        tf.keras.layers.Dense(path_len, name='dense_relu_1'),
        tf.keras.layers.LeakyReLU(),
            #kernel_regularizer=tf.keras.regularizers.l1(l1=0.001)),
        tf.keras.layers.Dense(path_len, name='dense_linear_1'),
            #kernel_regularizer=tf.keras.regularizers.l2(l2=0.001)),
        tf.keras.layers.Reshape((1, path_len)),
    ]

    grouped_postprocess_layers = [
        [ tf.keras.layers.Activation(tf.math.sin, name='x_pos') ],
        [ tf.keras.layers.Activation(tf.math.cos, name='y_pos') ],
    ]

    ## Assemble model
    if model_name is None:
        model_name = f"b{batch_size}_{loss_function}_{optimizer}{learning_rate:.02e}"

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
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    log_dir = os.path.join(output_dir, 'logs', f'{timestamp}_{model_name}')
    checkpoint_freq = 'epoch' if train_steps_per_epoch is None else train_steps_per_epoch*checkpoint_period
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_freq=checkpoint_freq),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=500)]

    save_path = os.path.join(output_dir, 'models', f'{timestamp}_{model_name}')
    if vis_model:
        tf.keras.utils.plot_model(model, to_file=save_path+'.png', show_shapes=True)
    print(model.summary())

    model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, batch_size=batch_size, epochs=epochs,
        steps_per_epoch=train_steps_per_epoch, validation_steps=val_steps)

    # Save model
    model.save(save_path+'.tf', include_optimizer=False)
