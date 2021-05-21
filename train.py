import datetime, os
from typing import Iterable
import tensorflow as tf

def get_proto_parser(path_len:int, n_pins:int):
    @tf.function
    def parser(proto):
        example = tf.io.parse_single_example(proto, {
            'image/pixels': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/size': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'target/path': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'target/mask': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'target/compressed': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'target/length': tf.io.FixedLenFeature([], tf.int64, default_value=0)
        })

        #pattern = tf.io.parse_tensor(example['target/path'], tf.float32)
        #target_pdf = mask_pdf(pattern[1:path_len], pattern[2:path_len+1])
        target_pdf = tf.io.parse_tensor(example['target/mask'], tf.float32)

        pixels = tf.io.parse_tensor(example['image/pixels'], tf.float32)
        res = example['image/size']
        ex_input = tf.reshape(pixels, (res,res,1))

        #compressed_mask = tf.io.parse_tensor(example['target/compressed'], tf.float32)
        #N = example['target/length']
        #ex_output = tf.reshape(compressed_mask, (N, -1))
        return [ex_input, target_pdf]
        #return [ex_input, ex_output]

    return parser

class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

def do_train(train_records:Iterable[str], res:int, path_len:int, output_dir:str,
            name:str=None, n_pins:int=300, n_cons:int=10, batch_size:int=10, epochs:int=1,
            overshoot_epochs:int=30, random_seed:int=42) -> None:
    tf.random.set_seed(random_seed)
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    ## Assemble model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(res,res,1)))
    # model.add(tf.keras.layers.Conv2D(2, 1))
    # model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2500, activation='relu', name='dense_relu_01'))
    # model.add(tf.keras.layers.Dense(2500, activation='relu', name='dense_relu_02'))
    # model.add(tf.keras.layers.Dense(2500, activation='relu', name='dense_relu_03'))
    # model.add(tf.keras.layers.Dense(2500, activation='relu', name='dense_relu_04'))
    # model.add(tf.keras.layers.Dense(2500, activation='relu', name='dense_relu_05'))
    model.add(tf.keras.layers.Dense(n_pins**2, activation='linear', name='output'))
    #model.add(tf.keras.layers.Dense(path_len, activation=tf.math.sin, name='dense_sine'))
    #model.add(tf.keras.layers.Dense(path_len, activation='linear', name='dense_linear'))
    #model.add(tf.keras.layers.experimental.preprocessing.Rescaling(n_pins))

    ## Load dataset
    parser = get_proto_parser(path_len, n_pins)

    raw_dataset_train = tf.data.TFRecordDataset(train_records)
    dataset_train = raw_dataset_train.map(parser).shuffle(batch_size*5)
    batched_train = dataset_train.batch(batch_size, drop_remainder=True)

    ## Train model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-1)
    #dtheta = utils.d_theta_loss(n_pins)
    #mask_pdf_loss = utils.get_mask_pdf_loss(n_pins)
    #model.compile(optimizer=optimizer, loss=dtheta, metrics=[mask_pdf_loss])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=overshoot_epochs, restore_best_weights=True),
                tf.keras.callbacks.TerminateOnNaN(),
                # tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/latest.tf', monitor='loss', save_best_only=True),
                CustomTensorBoard(log_dir=os.path.join(output_dir, 'logs', f'{timestamp}_{name}'))]

    model.fit(batched_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    model.save(os.path.join(output_dir, 'models', f'{timestamp}_{name}.final.tf'))
