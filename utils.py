import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

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

def get_loss_function(n_pins):
    @tf.function
    def index_error(y_true, y_pred):
        y_true_norm = 2 * np.pi / n_pins * y_true
        y_pred_norm = 2 * np.pi / n_pins * y_pred
        x0, y0 = tf.sin(y_true_norm), tf.cos(y_true_norm)
        x1, y1 = tf.sin(y_pred_norm), tf.cos(y_pred_norm)
        sq_err = (x1-x0)**2 + (y1-y0)**2
        return tf.reduce_mean(sq_err, axis=-1)

    return index_error
