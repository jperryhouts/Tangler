import os, random, logging
from typing import Callable, Union
from pathlib import Path
from subprocess import Popen, PIPE
from io import BytesIO
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

try:
    from PIL import Image, ImageOps
except:
    print("Unable to load PIL package")

try:
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-talk')
except:
    print("Unable to load pyplot")

def load_img(src: str, res: int, grayscale:bool=True) -> np.array:
    img = Image.open(src)
    if grayscale:
        img = ImageOps.grayscale(img)

    crop = None
    if img.size[0] != img.size[1]:
        cx, cy = (img.size[0]//2, img.size[1]//2)
        size2 = min(img.size)//2
        crop = (cx-size2, cy-size2, cx+size2, cy+size2)

    if res != -1:
        if (img.size[0] != res) or (crop is not None):
            img = img.resize((res, res), box=crop)

    return np.array(img)

def img_to_ravel(img:np.ndarray) -> np.ndarray:
    res = img.shape[0]
    sp = Popen(['/home/jmp/bin/raveler','-r',str(res),'-f','tsv','-'], stdout=PIPE, stdin=PIPE)
    img = img.astype(np.uint8)
    so = sp.communicate(input=img.tobytes())
    pins = np.loadtxt(BytesIO(so[0])).T[0]
    return pins.astype(np.float32)

@tf.function
def periodic_padding(tensor:tf.Tensor, pad:int) -> tf.Tensor:
    upper_pad = tensor[:,-pad:,:]
    lower_pad = tensor[:,:pad,:]
    partial = tf.concat([upper_pad, tensor, lower_pad], axis=1)
    left_pad = partial[:,:,-pad:]
    right_pad = partial[:,:,:pad]
    padded = tf.concat([left_pad, partial, right_pad], axis=2)
    return padded

@tf.function
def pooled_binary_crossentropy(y_true:tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
    y_true = periodic_padding(y_true, 1)
    y_pred = periodic_padding(y_pred, 1)
    y_true = tf.nn.max_pool2d(tf.expand_dims(y_true, -1), 3, 1, padding='VALID')
    y_pred = tf.nn.max_pool2d(tf.expand_dims(y_pred, -1), 3, 1, padding='VALID')
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
    return loss

def target_to_complex(target:np.ndarray, n_pins:int=256) -> np.ndarray:
    target = target.reshape((n_pins, n_pins, 2))
    return target[:,:,0] + 1j*target[:,:,1]

def target_to_tangle(target:np.ndarray, n_pins:int=256) -> np.ndarray:
    return target.reshape((n_pins,n_pins))
    # target_fft = target_to_complex(target, n_pins)
    # target_fft = np.fft.fftshift(target_fft)
    # tangle = np.fft.ifft2(target_fft, s=(n_pins,n_pins))
    # return tangle.real

def tangle_to_tftarget(tangle:tf.Tensor) -> tf.Tensor:
    return tangle
    # target = tf.cast(tangle, tf.complex64)
    # target = tf.signal.fft2d(target)
    # target = tf.signal.fftshift(target)
    # real = tf.math.real(target)
    # imag = tf.math.imag(target)
    # return tf.stack([real, imag], axis=-1)

def dropout(array:np.ndarray, rate:float) -> np.ndarray:
    mask = 1*(np.random.random(array.shape) > (1.0-rate))
    return mask*array

def resample(tangled:np.ndarray, threshold:float, n_dropouts:int=10) -> np.ndarray:
    resampled = np.zeros(tangled.shape)
    for T in np.linspace(threshold, tangled.max(), n_dropouts, endpoint=False):
        resampled += dropout(1.0*(tangled > T), 1/n_dropouts)
    return 1*(resampled > 0)

def untangle(tangled:np.ndarray, path_len:int, threshold:float=0.0,
                dtype:np.dtype=np.uint8) -> np.ndarray:

    locs = np.where(tangled > threshold)

    path = np.zeros(path_len, dtype=dtype)
    idx = 0
    for i in range(locs[0].size):
        path[idx] = locs[0][i]
        path[idx+1] = locs[1][i]
        idx += 2
        if idx >= path_len-1:
            logging.warn(f"String path buffer exceeded: {2*locs[0].size} > {path_len}")
            break

    return path

def plot_example(image:np.ndarray, target:np.ndarray, n_pins:int=256, fft:bool=False) -> None:
    # _, ax = plt.subplots((2 if fft else 1), 3, figsize=(14,8))
    # if not fft:
    #     ax = [ax]
    _, ax = plt.subplots(1, 2, figsize=(9,4))
    ax0, ax1 = ax[0], ax[1]

    ax0.imshow(image, aspect=1, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax0.set_axis_off()

    if fft:
        tangle_fft = target_to_complex(target)
        ax[1][0].imshow(np.log1p(tangle_fft.real**2), aspect=1, interpolation='nearest',
            extent=(-n_pins//2,n_pins//2,-n_pins//2,n_pins//2), cmap=plt.cm.gray)
        ax[1][0].set_title('Amplitude')
        ax[1][1].imshow(np.log1p(tangle_fft.imag**2), aspect=1, interpolation='nearest',
            extent=(-n_pins//2,n_pins//2,-n_pins//2,n_pins//2), cmap=plt.cm.gray)
        ax[1][1].set_title('Phase')

    tangle = target_to_tangle(target, n_pins)
    ax1.imshow(tangle, aspect=1, cmap=plt.cm.gray_r, interpolation='nearest')

    # w = np.where(np.tril(tangle) > 0.5)
    # path = []
    # for i in range(w[0].size):
    #     path.append(w[0][i])
    #     path.append(w[1][i])

    # print("Path length:", len(path))
    # raveled = np.array(path).astype(np.float)
    # theta = raveled*2*np.pi/n_pins

    # ax[0][1].plot(np.sin(theta), 1-np.cos(theta), 'k-', lw=0.01)
    # ax[0][1].set_aspect(1.0)
    # ax[0][1].set_axis_off()

    # ax[1][2].set_axis_off()
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
    C_MASK = C_MASK.reshape((res,res,1)).astype(np.uint8)
    C_MASK = tf.convert_to_tensor(C_MASK, dtype=tf.uint8)
    C_MASK_INV = 1-C_MASK

    for example in examples:
        image_data = tf.io.gfile.GFile(f"{example}.jpg", 'rb').read()
        img = tf.io.decode_jpeg(image_data)
        img = tf.reshape(img, (res, res, 1))
        img = img*C_MASK + 127*C_MASK_INV

        raveled = np.loadtxt(f"{example}.raveled", dtype=np.int64).flatten()
        tangle = np.zeros((n_pins, n_pins), dtype=np.int64)
        tangle[raveled[:-1],raveled[1:]] = 1
        tangle[raveled[1:],raveled[:-1]] = 1
        tangle = tf.convert_to_tensor(tangle, tf.int64)

        target = tangle_to_tftarget(tangle)
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
    C_MASK = C_MASK.reshape((res,res,1)).astype(np.uint8)
    C_MASK = tf.convert_to_tensor(C_MASK, dtype=tf.uint8)
    C_MASK_INV = 1-C_MASK

    @tf.function()
    def decode_example(serialized:tf.Tensor) -> tuple[tf.Tensor,tf.Tensor]:
        example = parse_example(serialized)

        img = tf.image.decode_jpeg(example['image/encoded'])
        img = tf.reshape(img, (res, res, 1))

        sparsity = example['target/sparsity']
        tangle_indices = tf.io.parse_tensor(example['target/indices'], tf.uint8)
        tangle_indices = tf.cast(tangle_indices, tf.int64)
        tangle_indices = tf.reshape(tangle_indices, (sparsity, 2))
        tangle_sp = tf.sparse.SparseTensor(tangle_indices, tf.ones(sparsity, tf.int64), (n_pins, n_pins))
        tangle = tf.sparse.to_dense(tangle_sp)
        tangle += tf.transpose(tangle)
        tangle = tf.clip_by_value(tangle, 0, 1)
        img = img*C_MASK + 127*C_MASK_INV

        ## Rotate example
        if rotate == 'any':
            rotate_pins = tf.random.uniform([], 0, n_pins, dtype=tf.int32)
            rotate_theta = 2 * np.pi * tf.cast(rotate_pins, tf.float32) / n_pins
            img = tfa.image.rotate(img, rotate_theta, fill_mode='constant', fill_value=127)
            tangle = tf.roll(tangle, shift=rotate_pins, axis=0)
            tangle = tf.roll(tangle, shift=rotate_pins, axis=1)
        elif rotate == '90':
            rotate_k = tf.random.uniform([], 0, 4, dtype=tf.int32)
            img = tf.image.rot90(img, k=rotate_k)
            tangle = tf.roll(tangle, shift=rotate_k*n_pins//4, axis=0)
            tangle = tf.roll(tangle, shift=rotate_k*n_pins//4, axis=1)

        target = tangle_to_tftarget(tangle)
        return (img, target)

    return decode_example
