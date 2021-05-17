import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

def load_img(src: str, res: int) -> np.array:
    img = Image.open(src) if type(src) is str else src
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
        ac = self.pin2coord(a)
        bc = self.pin2coord(b)
        return np.sqrt((ac[0]-bc[0])**2 + (ac[1]-bc[1])**2)

    def pin2coord(self, pin):
        theta = pin*2*np.pi/self.n_pins
        x = np.sin(theta)/2+1
        y = np.cos(theta)/2+1
        return (x, y)

    def idx2pair(self, idx):
        return self.idx2seg[idx]

    def pair2idx(self, a, b):
        a, b = sorted([a,b])
        return self.seg2idx[a][b-a]

    def t_path2mask_idx(self, a, b):
        a = tf.cast(a, dtype=tf.int32)
        b = tf.cast(b, dtype=tf.int32)
        a = tf.math.floormod(a, self.n_pins)
        b = tf.math.floormod(b, self.n_pins)

        tmp1 = tf.gather(self.t_seg2idx, a, axis=0)
        tmp2 = tf.gather(tmp1, b, axis=1)
        idx = tf.linalg.diag_part(tmp2)
        return idx

    @tf.custom_gradient
    def loss_function(self, y_true, layer_inputs):
        sigmoid_activation = layer_inputs

        # with tf.GradientTape() as tape:
        #     tape.watch(layer_inputs)
        #     sigmoid_activation = tf.math.sigmoid(layer_inputs)
        #     sigmoid_activation_gradient = tape.gradient(sigmoid_activation, layer_inputs)

        y_pred = self.n_pins*sigmoid_activation

        y_shape = (10, 1000)

        loss = np.zeros(y_shape[0])
        grad = np.zeros(y_shape)

        for i in range(y_shape[0]):
            a_true, b_true = (y_true[i][:-1], y_true[i][1:])
            a_pred, b_pred = (y_pred[i][:-1], y_pred[i][1:])

            true_idx = self.t_path2mask_idx(a_true, b_true)
            pred_idx = self.t_path2mask_idx(a_pred, b_pred)
            true_mask = tf.math.bincount(true_idx, minlength=self.size, dtype=tf.int32)
            pred_mask = tf.math.bincount(pred_idx, minlength=self.size, dtype=tf.int32)
            err_mask = true_mask-pred_mask
            err = self.t_lengths * tf.cast(tf.square(err_mask), dtype=tf.float32)

            loss[i] = tf.reduce_sum(err) / self.size

            pred_idx1 = self.t_path2mask_idx(a_pred, b_pred+1)
            pred_mask1 = tf.math.bincount(pred_idx1, minlength=self.size, dtype=tf.int32)
            err_mask1 = pred_mask1 - true_mask
            err1 = self.t_lengths * tf.cast(tf.square(err_mask1), dtype=tf.float32)

            derr = tf.gather((err1-err), pred_idx1)
            #grad[i] = sigmoid_activation_gradient[i] * tf.pad(derr, [[1, 0]])
            grad[i] = tf.pad(derr, [[1, 0]])

        t_grad = tf.convert_to_tensor(grad, dtype=tf.float32)

        def gradient(y):
            y1 = tf.cast(y, dtype=tf.float32)
            g = tf.transpose(tf.transpose(t_grad) * y1)
            return (None, g)

        return loss, gradient