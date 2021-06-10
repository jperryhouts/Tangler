import os, random
from pathlib import Path
from typing import Callable
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

try:
    from PIL import Image, ImageOps
except:
    print("Unable to load PIL package")

try:
    import matplotlib.pyplot as plt
except:
    print("Unable to load pyplot")

def load_img(src: str, res: int) -> np.array:
    img = Image.open(src)
    img = ImageOps.grayscale(img)

    crop = None
    if img.size[0] != img.size[1]:
        cx, cy = (img.size[0]//2, img.size[1]//2)
        size2 = min(img.size)//2
        crop = (cx-size2, cy-size2, cx+size2, cy+size2)

    if (img.size[0] != res) or (crop is not None):
        img = img.resize((res, res), box=crop)

    return np.array(img)

def plot_example(image:np.ndarray, target:np.ndarray, n_pins:int=256) -> None:
    _, ax = plt.subplots(1, 3, figsize=(14,4))
    ax[0].imshow(image, aspect=1, cmap=plt.cm.gray, vmin=-1, vmax=1)
    ax[1].imshow(target, aspect=1, cmap=plt.cm.gray_r, interpolation='nearest')

    target = np.tril(target)
    w = np.where(target > 0)
    path = []
    for i in range(w[0].size):
        path.append(w[0][i])
        path.append(w[1][i])

    print("Path length:", len(path))
    raveled = np.array(path).astype(np.float)
    theta = raveled*2*np.pi/n_pins

    ax[2].plot(np.sin(theta), 1-np.cos(theta), 'k-', lw=0.01)
    ax[2].set_axis_off()
    plt.show()

def example_generator(src_dir:str, n_pins:int, res:int=-1) -> tuple[tf.Tensor, tf.Tensor]:
    if type(src_dir) is bytes:
        src_dir = src_dir.decode('utf-8')
    assert os.path.isdir(src_dir)
    get_stem = lambda p: str(os.path.join(p.parent.absolute(), p.stem))
    examples = [get_stem(path) for path in Path(src_dir).rglob('*.jpg')]
    assert len(examples) > 0
    random.shuffle(examples)

    X,Y = np.mgrid[-1:1:res*1j,-1:1:res*1j]
    R2 = X**2+Y**2
    C_MASK = 1*(R2<1.0)
    C_MASK = C_MASK.reshape((res,res,1)).astype(np.float32)
    C_MASK = tf.convert_to_tensor(C_MASK, dtype=tf.float32)

    for example in examples:
        image_data = tf.io.gfile.GFile(f"{example}.jpg", 'rb').read()
        img = tf.io.decode_jpeg(image_data)
        img = tf.reshape(img, (res, res, 1))

        raveled = np.loadtxt(f"{example}.raveled", dtype=np.int64).flatten()
        target = np.zeros((n_pins, n_pins), dtype=np.int64)
        target[raveled[:-1],raveled[1:]] = 1
        target[raveled[1:],raveled[:-1]] = 1
        target = tf.convert_to_tensor(target, tf.int64)

        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img *= C_MASK
        return (img, target)

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

def get_decoder(res:int, n_pins:int=256, rotate:str='none'
        ) -> Callable[[tf.Tensor],tuple[tf.Tensor,tf.Tensor]]:
    assert rotate in ['none', '90', 'any']

    X,Y = np.mgrid[-1:1:res*1j,-1:1:res*1j]
    R2 = X**2+Y**2
    C_MASK = 1*(R2<1.0)
    C_MASK = C_MASK.reshape((res,res,1)).astype(np.float32)
    C_MASK = tf.convert_to_tensor(C_MASK, dtype=tf.float32)

    @tf.function()
    def decode_example(serialized:tf.Tensor) -> tuple[tf.Tensor,tf.Tensor]:
        example = parse_example(serialized)

        img = tf.image.decode_jpeg(example['image/encoded'])
        img = tf.reshape(img, (res, res, 1))

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

        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img *= C_MASK
        return (img, target)

    return decode_example
