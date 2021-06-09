import os, random, logging
from pathlib import Path
import itertools
import re
from typing import Tuple, Any, Iterable
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import TangledModel

try:
    from PIL import Image, ImageOps
except:
    print("Unable to load PIL package")

try:
    import cv2
except:
    print("Unable to load OpenCV package")

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

    if (img.size[0] != res) or (crop is not None):
        img = img.resize((res, res), box=crop)

    return np.array(img)

def pin_path_to_target_v4(path:Iterable, n_pins:int) -> np.ndarray:
    target = np.zeros((n_pins, n_pins, 1), dtype=np.float16)
    for a,b in zip(path[:-1], path[1:]):
        target[a][b][0] += 1
        target[b][a][0] += 1
    # zero the elements near the diagonal
    target = np.tril(target,-15)+np.triu(target,15)
    return target

def example_generator(src_dir:str, n_pins:int, res:int=-1) -> Tuple[tf.Tensor, tf.Tensor]:
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
    C_MASK_INV = tf.convert_to_tensor(1-C_MASK, dtype=tf.uint8)

    for example in examples:
        image_data = tf.io.gfile.GFile(f"{example}.jpg", 'rb').read()
        image = tf.io.decode_jpeg(image_data)
        image = C_MASK*image + 127*C_MASK_INV

        raveled = np.loadtxt(f"{example}.raveled", dtype=np.int64).flatten()
        # indices = set([tuple(sorted(ij)) for ij in zip(raveled[:-1], raveled[1:])])
        # indices = np.array(list(indices))
        # print(indices)c
        # indices = np.array([raveled[:-1], raveled[1:]]).T
        #print(indices)
        #indices = np.sort(indices, axis=-1)
        #print(indices)
        # tf.convert_to_tensor([x, y])
        # target_sp = tf.SparseTensor(indices, tf.ones(indices.shape[0]), (n_pins, n_pins))
        # target_sp = tf.sparse.reorder(target_sp)
        # target = tf.sparse.to_dense(target_sp)
        target = np.zeros((n_pins, n_pins), dtype=np.int64)
        target[raveled[:-1],raveled[1:]] = 1
        target[raveled[1:],raveled[:-1]] = 1
        ## Remove diagonal?
        # target = np.tril(target,-1)+np.triu(target,1)
        target = tf.convert_to_tensor(target, tf.int64)

        yield (image, target)

def plot_example(image:np.ndarray, target:np.ndarray, n_pins:int=256) -> None:
    _, ax = plt.subplots(1, 3, figsize=(14,4))
    ax[0].imshow(image, aspect=1, cmap=plt.cm.gray, vmin=0, vmax=255)
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
            'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'target/rows': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'target/cols': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'target/value': tf.io.FixedLenFeature([], tf.string, default_value=''),
        })

    res = example['image/res']
    image = tf.image.decode_jpeg(example['image/encoded'])
    image = tf.cast(image, tf.float32)
    input_data = tf.reshape(image, (res, res, 1))

    n_pins = example['target/rows']
    n_cols = example['target/cols']
    target = tf.io.parse_tensor(example['target/value'], tf.float32)
    target_data = tf.reshape(target, (2, n_pins, n_cols))

    return (input_data, target_data)

def pin_path_to_target(path:Iterable, n_pins:int, n_cons:int) -> np.ndarray:
    path = path.astype(np.int)
    res = [set() for _ in range(n_pins)]

    for i in range(len(path)-1):
        a, b = path[i:i+2]

        if len(res[a]) < n_cons:
            res[a].add(b)

        if len(res[b]) < n_cons:
            res[b].add(a)

    for i in range(len(res)):
        ## Sort columns by distance from pin i
        r_i = np.array(list(res[i]))
        #R = abs((abs(r_i-i)+n_pins//2)%n_pins-n_pins//2)
        #Rr = sorted(zip(R, r_i), key=lambda x: x[0], reverse=True)
        #res[i] = [x[1] for x in Rr]

        res[i] = sorted((r_i-i)%n_pins, reverse=True)

        while len(res[i]) < n_cons:
            res[i].append(1)

    theta = 2 * np.pi * np.array(res) / n_pins
    res = np.array([np.sin(theta), np.cos(theta)])
    return res

def theta_matrix_to_pin_path(thetas:np.ndarray, n_pins:int, max_len:int) -> np.ndarray:
    ppins = (n_pins*thetas/(2*np.pi)).astype(np.int)
    for i in range(n_pins):
        ppins[i] = (ppins[i]+i)%n_pins
    ppins = ppins.tolist()

    n_empty = 0

    path_len = 1
    path = np.zeros(max_len, dtype=np.int)
    while n_empty < n_pins and path_len < max_len:
        prev_pin = path[path_len-1]
        options = ppins[prev_pin]

        was_non_empty = (len(options) > 0)

        next_pin = options.pop(0) if len(options) > 0 else (prev_pin+1)%n_pins
        if next_pin == prev_pin:
            next_pin = options.pop(0) if len(options) > 0 else (prev_pin+1)%n_pins

        is_now_empty = (len(options) == 0)

        if was_non_empty and is_now_empty:
            n_empty += 1

        path[path_len] = next_pin

        path_len += 1

    # print("Path length:",path_len)

    return path

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

def inference_to_path(result:np.ndarray, path_len:int, threshold:float=0.0) -> np.ndarray:
    rmin = result.min()
    res = np.tril(result-rmin)+rmin
    w = np.where(res > threshold)

    path = np.zeros(path_len)
    idx = 0
    for i in range(w[0].size):
        path[idx] = w[0][i]
        path[idx+1] = w[1][i]
        idx += 2
        if idx >= path_len-1:
            logging.warn(f"String path buffer exceeded: {2*w[0].size} > {path_len}")
            break

    # print(idx)
    return path

class TangledPredictor():
    def __init__(self, model_path:str) -> None:
        base_model = TangledModel()
        base_model.load_weights(model_path)
        self.model = base_model
        # self.model = tf.keras.Model(base_model.input, base_model.layers[-2].output)
        self.res, self.n_pins = base_model.res, base_model.n_pins
        self.path_len = 60000
        self.prediction_num = 0

    def show(self, result, threshold=0.0):
        _, ax = plt.subplots(1,1)
        vrange = max(abs(result.max()), abs(result.min()))
        im = ax.imshow(result, aspect='auto', cmap=plt.cm.seismic, interpolation='nearest', vmin=-vrange, vmax=vrange)
        ax.contour(result, levels=[threshold], colors='k', linewidths=0.5)
        plt.colorbar(im, ax=ax)
        plt.show()

    def predict_convert(self, inputs:np.ndarray) -> np.ndarray:
        img = inputs.reshape((1,self.res,self.res,1))
        n_pins = self.n_pins
        result = self.model.predict(img.astype(np.uint8))
        result = result.astype(np.float).reshape((n_pins,n_pins))
        #result = result.astype(np.float).reshape((2*n_pins, 2*n_pins))
        #a = n_pins//2
        #b = a + n_pins
        #result = result[a:b,a:b]

        rmin = result.min()
        res = result - rmin
        res = np.tril(res, -2) + np.triu(res, -2)
        res += rmin
        res = (res + res.T) / 2.0
        threshold = np.percentile(res, 60)
        # threshold = 0.0

        self.prediction_num += 1
        if self.prediction_num == 10:
            print('result range',result.min(), result.max())
            print('threshold',threshold)
            print(list(range(5,100,5)))
            print(np.percentile(res, range(1,100,5)))
            self.show(res, threshold)

        return inference_to_path(res, self.path_len, threshold)

class FlatModel(tf.keras.Model):
    def __init__(self, model_path:str, n_pins:int=256) -> None:
        base_model = tf.keras.models.load_model(model_path)
        super().__init__(base_model.input, base_model.layers[-4].output)
        self.res = self.input.type_spec.shape[1]
        self.path_len = self.output.type_spec.shape[-1]
        self.n_pins = n_pins

    def predict(self, inputs:np.ndarray) -> np.ndarray:
        img = inputs.astype(np.float32).reshape((1,self.res,self.res,1))
        thetas = super().predict(img)[0][0]
        return thetas

class Camera():
    def __init__(self, model_res:int, capture_source:Any=0):
        self.camera = cv2.VideoCapture(capture_source)
        self.load_crop_dimensions()
        self.model_res = model_res
        self.load_cmask()

    def close(self):
        print('Releasing camera')
        self.camera.release()

    def load_crop_dimensions(self):
        w, h = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == h:
            self.y0, self.x0 = [0,0]
            self.y1, self.x1 = [h-1,w-1]
        else:
            imres = min(w,h)
            wc, hc = w//2, h//2
            self.y0, self.x0 = (hc-imres//2, wc-imres//2)
            self.y1, self.x1 = (hc+imres//2, wc+imres//2)

    def load_cmask(self):
        res = self.model_res
        X,Y = np.mgrid[-1:1:res*1j,-1:1:res*1j]
        R2 = X**2+Y**2
        self.cmask = (1*(R2<1.0)).astype(np.uint8)
        self.cmask_inv = 1-self.cmask

    def img_stream(self):
        while self.camera.isOpened():
            success, frame = self.camera.read()

            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = img[self.y0:self.y1,self.x0:self.x1]
                img = cv2.resize(img, (self.model_res, self.model_res))
                img *= self.cmask
                img += 127*self.cmask_inv
                yield img

class ImageIterator():
    def __init__(self, source:Any, cycle:bool, res:int) -> None:
        if source == 'webcam':
            self.type = 'webcam'
            self.cam = Camera(res)
            self.source = self.cam.img_stream()
        else:
            self.type = 'files'
            self.paths = itertools.cycle(source) if cycle else itertools.chain(source)
            def gen():
                for path in self.paths:
                    yield load_img(path, res)
            self.source = gen()

    def close(self) -> None:
        print("Closing ImageIterator")
        if self.type == 'webcam':
            self.cam.close()

    def __next__(self):
        return self.source.__next__()