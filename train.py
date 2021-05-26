import datetime, os, sys
from typing import Iterable
import tensorflow as tf

import utils

def get_data_shape(ds):
    inputs, outputs = ds.take(1).map(utils.decode_example).as_numpy_iterator().next()
    return (inputs.shape, outputs.shape)

def do_train(train_records:Iterable[str], val_records:Iterable[str], output_dir:str,
            model_name:str=None, checkpoint_path:str='/tmp/latest.tf', checkpoint_period:int=1,
            loss_function:str='mse', optimizer:str='adam', learning_rate:float=1e-3,
            weighted_loss:bool=False, batch_size:int=100, use_mixed_precision:bool=False,
            epochs:int=1, patience:int=30, train_steps_per_epoch:int=2000, val_steps:int=200) -> None:
    tf.random.set_seed(42)

    if use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    ## Load data
    raw_dataset_train = tf.data.TFRecordDataset(train_records)
    ((res,_,_), (_, n_pins, n_cons)) = get_data_shape(raw_dataset_train)
    ds_train = raw_dataset_train.map(utils.decode_example).shuffle(5*batch_size).repeat()
    ds_train = ds_train.batch(batch_size, drop_remainder=True)

    raw_dataset_val = tf.data.TFRecordDataset(val_records)
    ds_val = raw_dataset_val.map(utils.decode_example).shuffle(5*batch_size).repeat()
    ds_val = ds_val.batch(batch_size, drop_remainder=True)

    ## Define model
    preprocess_layers = [
        tf.keras.layers.experimental.preprocessing.Rescaling(-1./255, offset=1.0, name='scale_invert'),
    ]

    grouped_convolutional_layers = [
        [
            tf.keras.layers.Flatten(name='bypass'),
        ],
        # [
        #     tf.keras.layers.Conv2D(15, 3, padding='valid'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
        #     tf.keras.layers.Flatten(name='conv_15'),
        # ],
        # [
        #     tf.keras.layers.Conv2D(30, 5, padding='valid'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
        #     tf.keras.layers.Flatten(name='conv_30'),
        # ],
    ]

    hidden_layers = [
        tf.keras.layers.Dense(n_pins*n_cons//2+1, name='dense_relu_1', activation='relu'),
            # kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(n_pins*n_cons, name='wide_linear'),
        tf.keras.layers.Reshape((1, n_pins, n_cons)),
    ]

    grouped_postprocess_layers = [
        [ tf.keras.layers.Activation(tf.math.sin, name='x_pos') ],
        [ tf.keras.layers.Activation(tf.math.cos, name='y_pos') ],
    ]

    ## Assemble model
    if model_name is None:
        model_name = f"r{res}_k{n_pins}_c{n_cons}_b{batch_size}_{loss_function}_{optimizer}{learning_rate:.02e}"

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

    if weighted_loss:
        xw = tf.stack([tf.range(0,n_cons,dtype=tf.float32) for _ in range(n_pins)])
        yw = tf.stack([tf.range(0,n_cons,dtype=tf.float32) for _ in range(n_pins)])
        loss_weights = 1.0 / (1.0 + (1.0/n_cons)*tf.stack([xw, yw]))
        loss_norm = 2 * n_cons * n_pins / tf.reduce_sum(loss_weights)
        loss_weights = loss_norm * loss_weights
        # Regularization fails unless loss_weights is a list. See
        # https://stackoverflow.com/questions/65970626/regularizer-causes-valueerror-shapes-must-be-equal-rank
        loss_weights= loss_weights.numpy().tolist()
    else:
        loss_weights = None

    model.compile(optimizer=opt, loss=loss_function, loss_weights=loss_weights)

    ## Train model
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    checkpoint_freq = train_steps_per_epoch*checkpoint_period
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_freq=checkpoint_freq),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'logs', f'{timestamp}_{model_name}'))]

    save_path = os.path.join(output_dir, 'models', f'{timestamp}_{model_name}')
    tf.keras.utils.plot_model(model, to_file=save_path+'.png', show_shapes=True)
    print(model.summary())

    model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, batch_size=batch_size, epochs=epochs,
        steps_per_epoch=train_steps_per_epoch, validation_steps=val_steps)

    # Save model
    model.save(save_path+'.tf', include_optimizer=False)
