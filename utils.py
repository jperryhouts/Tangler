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

class Masks():
    def __init__(self, n_pins):
        self.n_pins = n_pins

        self.pin_coords = []
        for pin in range(n_pins):
            theta = pin*2*np.pi/n_pins
            x = np.sin(theta)/2+1
            y = np.cos(theta)/2+1
            self.pin_coords.append([x,y])
        self.pin_coords = np.array(self.pin_coords)

        self.seg2idx = np.zeros((n_pins,n_pins), dtype=np.int)
        self.idx2seg = []
        self.lengths = []
        self.size = 0
        for i in range(n_pins):
            for j in range(i,n_pins):
                self.seg2idx[i][j] = self.size
                self.seg2idx[j][i] = self.size
                self.idx2seg.append((i,j))
                self.lengths.append(self.get_length(i,j))
                self.size += 1
        self.lengths = 5*np.array(self.lengths)+1.0
        self.t_seg2idx = tf.convert_to_tensor(self.seg2idx, dtype=tf.int32)
        self.t_lengths = tf.convert_to_tensor(self.lengths, dtype=tf.float32)

    def get_length(self, a, b):
        dx = self.pin_coords[a] - self.pin_coords[b]
        return np.sqrt((dx**2).sum())

    def pin2coord(self, pin):
        return self.pin_coords[pin]

    def idx2pair(self, idx):
        return self.idx2seg[idx]

    def pair2idx(self, a, b):
        a, b = sorted([a,b])
        return self.seg2idx[a][b-a]

    ## a,b(10,999) -> idx(10,999)
    def t_path2mask_idx(self, a, b):
        a = tf.math.floormod(tf.cast(a, dtype=tf.int32), self.n_pins)
        b = tf.math.floormod(tf.cast(b, dtype=tf.int32), self.n_pins)
        tmp1 = tf.gather(self.t_seg2idx, a, axis=0, batch_dims=1)
        tmp2 = tf.gather(tmp1, b, axis=2, batch_dims=1)
        idx = tf.linalg.diag_part(tmp2)
        return idx

    @tf.custom_gradient
    def loss_function(self, y_true, y_pred):
        a_true, b_true = (y_true[:,:-1], y_true[:,1:])
        a_pred, b_pred = (y_pred[:,:-1], y_pred[:,1:])

        true_idx = self.t_path2mask_idx(a_true, b_true)
        pred_idx = self.t_path2mask_idx(a_pred, b_pred)
        true_mask = tf.math.bincount(true_idx, minlength=self.size, dtype=tf.int32, axis=-1)
        pred_mask = tf.math.bincount(pred_idx, minlength=self.size, dtype=tf.int32, axis=-1)
        err_mask = true_mask-pred_mask
        err = tf.cast(tf.square(err_mask), dtype=tf.float32) #* self.t_lengths

        loss = tf.reduce_sum(err, axis=-1)
        norm = 1.0 / tf.reduce_sum(0 * err + 1, axis=-1)
        loss = tf.transpose(norm * tf.transpose(loss))

        pred_idx1 = self.t_path2mask_idx(a_pred, b_pred+1)
        pred_mask1 = tf.math.bincount(pred_idx1, minlength=self.size, dtype=tf.int32, axis=-1)
        err_mask1 = pred_mask1 - true_mask
        err1 = tf.cast(tf.square(err_mask1), dtype=tf.float32) #* self.t_lengths

        derr = tf.gather((err1-err), pred_idx1, batch_dims=1)
        t_grad = tf.pad(derr, [[0,0], [1, 0]])

        def gradient(y):
            y1 = tf.cast(y, dtype=tf.float32)
            g = tf.transpose(tf.transpose(t_grad) * y1)
            return (None, g)

        return loss, gradient
