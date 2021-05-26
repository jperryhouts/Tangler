from typing import Tuple
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

def _int_to_feature(value: int) -> tf.train.Feature:
    int64_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int64_list)

def _tensor_to_feature(tensor: tf.Tensor) -> tf.train.Feature:
    serialized = tf.io.serialize_tensor(tensor)
    bytes_list = tf.train.BytesList(value=[serialized.numpy()])
    return tf.train.Feature(bytes_list=bytes_list)

def load_img(src: str, res: int) -> np.array:
    img = Image.open(src)
    img = ImageOps.grayscale(img)

    crop = None
    if img.size[0] != img.size[1]:
        cx, cy = (img.size[0]//2, img.size[1]//2)
        size2 = min(img.size)//2
        crop = (cx-size2, cy-size2, cx+size2, cy+size2)
    img = img.resize((res, res), box=crop)
    return np.array(img)

def encode_example(image:np.ndarray, target:np.ndarray) -> tf.train.Example:
    _, n_pins, n_cons = target.shape
    res = image.shape[0]

    image = tf.convert_to_tensor(image, dtype=tf.uint8)
    image = tf.image.encode_jpeg(image, format='grayscale')
    target = tf.convert_to_tensor(target, dtype=tf.float32)

    return tf.train.Example(features=tf.train.Features(feature={
        'image/res': _int_to_feature(res),
        'image/encoded': _tensor_to_feature(image),
        'target/rows': _int_to_feature(n_pins),
        'target/cols': _int_to_feature(n_cons),
        'target/value': _tensor_to_feature(target),
    }))

@tf.function
def decode_example(serialized:tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    example = tf.io.parse_single_example(serialized, {
            'image/res': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'target/rows': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'target/cols': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'target/value': tf.io.FixedLenFeature([], tf.string, default_value=''),
        })

    res = example['image/res']
    image = tf.image.decode_jpeg(example['image/encoded'])
    image = tf.reshape(image, (res, res, 1))

    n_pins = example['target/rows']
    n_cols = example['target/cols']
    target = tf.io.parse_tensor(example['target/value'], tf.float32)
    target = tf.reshape(target, (2, n_pins, n_cols))

    return [image, target]

def pin_path_to_target(path:np.ndarray, n_pins:int, n_cons:int) -> np.ndarray:
    path = path.astype(np.int)
    res = [set() for _ in range(n_pins)]

    for i in range(len(path)-1):
        a, b = path[i:i+2]

        if len(res[a]) < n_cons:
            res[a].add(b)

        if len(res[b]) < n_cons:
            res[b].add(a)

    for i in range(len(res)):
        res[i] = list(res[i])
        while len(res[i]) < n_cons:
            res[i].append(i+1)

    theta = 2 * np.pi * np.array(res) / n_pins
    res = np.array([np.sin(theta), np.cos(theta)])
    return res

def theta_matrix_to_pin_path(thetas:np.ndarray, n_pins:int, max_len:int) -> np.ndarray:
    ppins = (n_pins*thetas/(2*np.pi)).astype(np.int)%n_pins
    ppins = ppins.tolist()

    n_empty = 0

    pins = [0]
    while n_empty < n_pins and len(pins) < max_len:
        prev_pin = pins[-1]
        options = ppins[prev_pin]

        was_non_empty = (len(options) > 0)

        next_pin = options.pop(0) if len(options) > 0 else (prev_pin+1)%n_pins
        if next_pin == prev_pin:
            next_pin = options.pop(0) if len(options) > 0 else (prev_pin+1)%n_pins

        is_now_empty = (len(options) == 0)

        if was_non_empty and is_now_empty:
            n_empty += 1

        pins.append(next_pin)

    return np.array(pins)

class Mapping():
    def __init__(self, n_pins:int) -> None:
        self.n_pins = n_pins
        self.pin_coords = self.get_pin_mapping(n_pins)

    def get_pin_mapping(self, n_pins:int) -> np.ndarray:
        thetas = np.arange(n_pins) * 2 * np.pi / n_pins
        coords = np.zeros((n_pins, 2))
        coords[:,0] = np.sin(thetas)
        coords[:,1] = np.cos(thetas)
        return coords

    def pins2xy(self, pins:np.ndarray) -> np.ndarray:
        return self.pin_coords[pins].T

class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
