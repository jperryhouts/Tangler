import glob, os
import tensorflow as tf

from utils import Masks

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

def do_train(datadir: str, res: int, path_len: int, n_pins: int=300, batch_size: int=10) -> tf.keras.Model:
    ## Assemble model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(res,res,1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(path_len, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(path_len, activation='linear'))

    ## Load dataset
    train_fnames = glob.glob(os.path.join(datadir, 'train', f'res={res}*.tfrecord'))
    assert len(train_fnames) > 0, 'No tfrecord dataset found.'
    raw_dataset_train = tf.data.TFRecordDataset(train_fnames)
    parser = get_proto_parser(path_len)
    dataset_train = raw_dataset_train.map(parser).shuffle(batch_size*5)
    batched_train = dataset_train.batch(batch_size, drop_remainder=True)

    ## Train model
    masks = Masks(n_pins)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=masks.loss_function)
    model.fit(batched_train, epochs=1, batch_size=batch_size)
    return model
