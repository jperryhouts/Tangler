import datetime, os, sys
from typing import Iterable
import tensorflow as tf
from numpy import pi
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.ops.gen_math_ops import Sin

@tf.function
def proto_parser(proto):
    example = tf.io.parse_single_example(proto, {
        'image/pixels': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/size': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        # 'target/path': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target/mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target/compressed': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target/length': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'target/npins': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'target/cons': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    })

    pixels = tf.io.parse_tensor(example['image/pixels'], tf.float32)
    res = example['image/size']
    ex_input = tf.reshape(pixels, (res,res,1))

    compressed_mask = tf.io.parse_tensor(example['target/compressed'], tf.float32)
    #N = example['target/length']
    n_pins = example['target/npins']
    n_cons = example['target/cons']
    # print(n_cons)
    # print(compressed_mask)
    ex_output = tf.reshape(compressed_mask, (2, n_pins, n_cons))

    return [ex_input, ex_output]

class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

def get_data_shape(ds):
    inputs, outputs = ds.take(1).map(proto_parser).as_numpy_iterator().next()
    return (inputs.shape, outputs.shape)

def do_train(train_records:Iterable[str], output_dir:str, model_name:str=None,
            loss:str='mse', optimizer:str='adam', learning_rate:float=1e-3,
            batch_size:int=10, epochs:int=1, overshoot_epochs:int=30,
            steps_per_epoch:int=1000, checkpoint_period=1, random_seed:int=42) -> None:
    tf.random.set_seed(random_seed)

    ## Load dataset
    raw_dataset_train = tf.data.TFRecordDataset(train_records)
    ((res,_,_), (_, n_pins, n_cons)) = get_data_shape(raw_dataset_train)
    dataset_train = raw_dataset_train.map(proto_parser).shuffle(10000).repeat()
    batched_train = dataset_train.batch(batch_size, drop_remainder=True)

    ## Define model
    preprocess_layers = [
        tf.keras.layers.experimental.preprocessing.Rescaling(-1./255, offset=1.0),
    ]

    grouped_convolutional_layers = [
        [ # bypass
            tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
            tf.keras.layers.Flatten(),
        ],
        [
            tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(5,5)),
            tf.keras.layers.Flatten(),
        ],
        [
            tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
            tf.keras.layers.Flatten(),
        ],
    ]

    hidden_layers = [
        tf.keras.layers.Dense(n_pins*n_cons//15, name='dense_1', activation='relu'),
        tf.keras.layers.Dense(n_pins*n_cons, name='dense_wide_1', activation='relu'),
        tf.keras.layers.Dense(n_pins*n_cons, name='dense_wide_2'),
        tf.keras.layers.Reshape((1, n_pins, n_cons)),
    ]

    grouped_postprocess_layers = [
        [ tf.keras.layers.Activation(tf.math.sin) ],
        [ tf.keras.layers.Activation(tf.math.cos) ],
    ]

    ## Assemble model
    if model_name is None:
        model_name = f"r{res}_k{n_pins}_c{n_cons}_b{batch_size}_{loss}_{optimizer}{learning_rate:g}"

    sequence = [tf.keras.Input(shape=(res,res,1))]

    for layer in preprocess_layers:
        sequence.append(layer(sequence[-1]))

    if len(grouped_convolutional_layers) > 0:
        convolutional_layers = []
        for group in grouped_convolutional_layers:
            sub_sequence = [group[0](sequence[-1])]
            for layer in group[1:]:
                sub_sequence.append(layer(sub_sequence[-1]))
            convolutional_layers.append(sub_sequence[-1])

        sequence.append(tf.keras.layers.Concatenate(axis=-1)(convolutional_layers))

    for layer in hidden_layers:
        sequence.append(layer(sequence[-1]))

    postprocess_layers = []
    for group in grouped_postprocess_layers:
        sub_sequence = [group[0](sequence[-1])]
        for layer in group[1:]:
            sub_sequence.append(layer(sub_sequence[-1]))
        postprocess_layers.append(sub_sequence[-1])

    sequence.append(tf.keras.layers.Concatenate(axis=1)(postprocess_layers))

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

    xw = tf.stack([tf.range(0,n_cons,dtype=tf.float32) for _ in range(n_pins)])
    yw = tf.stack([tf.range(0,n_cons,dtype=tf.float32) for _ in range(n_pins)])
    loss_weights = 1.0 / (1.0 + (1.0/n_cons)*tf.stack([xw, yw]))
    loss_norm = 2 * n_cons * n_pins / tf.reduce_sum(loss_weights)
    loss_weights = loss_norm * loss_weights

    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights.numpy())

    ## Train model
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    checkpoint_freq = steps_per_epoch*checkpoint_period
    callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=overshoot_epochs, restore_best_weights=True),
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/latest.tf', monitor='loss', save_freq=checkpoint_freq),
                CustomTensorBoard(log_dir=os.path.join(output_dir, 'logs', f'{timestamp}_{model_name}'))]

    save_path = os.path.join(output_dir, 'models', f'{timestamp}_{model_name}')
    tf.keras.utils.plot_model(model, to_file=save_path+'.png', show_shapes=True)
    print(model.summary())

    model.fit(batched_train, callbacks=callbacks, batch_size=batch_size,
        epochs=epochs, steps_per_epoch=steps_per_epoch)

    # Save model
    model.save(save_path+'.tf', include_optimizer=False)
