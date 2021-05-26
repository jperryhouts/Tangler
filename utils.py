import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow.keras.backend as K

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

def get_mask_pdf_loss(n_pins):
    def mask_pdf(a, b):
        x0 = a + b*n_pins
        x1 = b + a*n_pins
        x2 = tf.concat([x0, x1], axis=-1)
        xc = tf.math.bincount(x2, binary_output=True, axis=-1, minlength=2*n_pins, dtype=tf.float32)
        return xc

    @tf.function
    def mask_pdf_loss(y_true, y_pred):
        pdf_true = mask_pdf(y_true[:-1], y_true[1:])
        pdf_pred = mask_pdf(y_pred[:-1], y_pred[1:])
        loss = tf.keras.losses.binary_crossentropy(pdf_true, pdf_pred)

        # pdf_pred_1 = mask_pdf(y_pred[:-1], y_pred[1:]+1)
        # loss1 = tf.keras.losses.binary_crossentropy(pdf_true, pdf_pred_1)
        # grad = loss1-loss
        # print(grad.shape)
        # def gradient(y):
        #     g = tf.transpose(tf.transpose(grad), y)
        #     return (None, g)

        return loss

    return mask_pdf_loss

def get_mask_comparison_loss(n_pins):
    @tf.function
    def mask_comparison_loss(y_true, y_pred):
        x = y_true[:-1] + y_true[1:]*n_pins
        y = y_pred[:-1] + y_pred[1:]*n_pins
        x1 = tf.signal.fft(tf.cast(x, dtype=tf.complex64))
        y1 = tf.signal.fft(tf.cast(y, dtype=tf.complex64))
        conv1 = tf.multiply(x1, y1)
        conv = tf.signal.ifft(conv1)
        # conv2 = tf.cast(conv1, dtype=tf.float32)
        # coeff = tf.realdiv(1.0, conv2**2)
        # coeff = tf.keras.backend.conv1d(y_true, y_pred, padding='same')
        return tf.reduce_mean(-tf.abs(conv), axis=-1)

    return mask_comparison_loss

@tf.function
def convolution_loss(y_true, y_pred):
    x1 = tf.signal.fft(tf.cast(y_true, dtype=tf.complex64))
    y1 = tf.signal.fft(tf.cast(y_pred, dtype=tf.complex64))
    conv1 = tf.multiply(x1, y1)
    #conv = tf.signal.ifft(conv1)
    # conv2 = tf.cast(conv1, dtype=tf.float32)
    # coeff = tf.realdiv(1.0, conv2**2)
    # coeff = tf.keras.backend.conv1d(y_true, y_pred, padding='same')
    return tf.reduce_mean(-tf.abs(conv1), axis=-1)

@tf.function
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred #[:,:y_true.shape[1]]
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def d_theta_loss(n_pins):
    @tf.function
    def dtheta(y_true, y_pred):
        y_true_norm = y_true / n_pins
        y_pred_norm = y_pred / n_pins
        dy_abs = tf.abs(y_pred_norm - y_true_norm)
        dy_rel = n_pins*(tf.math.floormod(dy_abs+0.5, 1.0)-0.5)
        return tf.reduce_mean(tf.math.abs(dy_rel), axis=-1)

    return dtheta

def cartesian_loss(n_pins):
    @tf.function
    def error(y_true, y_pred):
        y_true_norm = 2 * np.pi / n_pins * y_true
        y_pred_norm = 2 * np.pi / n_pins * y_pred
        x0, y0 = tf.sin(y_true_norm), tf.cos(y_true_norm)
        x1, y1 = tf.sin(y_pred_norm), tf.cos(y_pred_norm)
        sq_err = (x1-x0)**2 + (y1-y0)**2
        return tf.reduce_mean(sq_err, axis=-1)

    return error
