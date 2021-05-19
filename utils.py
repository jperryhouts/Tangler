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

@tf.function
def lambda_activation(inputs):
    shape = (-1, 2, inputs.shape[-1]//2)
    tmp1 = tf.reshape(inputs, shape)
    tmp2 = tf.add(tmp1, 1e-6*tf.random.normal((2, shape[-1])))
    x = tmp2[:,0,:]
    y = tmp2[:,1,:]
    r = tf.sqrt(x**2 + y**2)
    rtanh = tf.math.tanh(r)
    #print(x.shape, y.shape, r.shape, inputs.shape)
    return inputs * tf.concat([rtanh, rtanh], axis=-1)

class RadialTanh(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RadialTanh, self).__init__(**kwargs)

    def call(self, inputs):
        shape = (-1, 2, inputs.shape[-1]//2)
        tmp1 = tf.reshape(inputs, shape)
        tmp2 = tf.add(tmp1, 1e-6*tf.random.normal((2, shape[-1])))
        x = tmp2[:,0,:]
        y = tmp2[:,1,:]
        r = tf.sqrt(x**2 + y**2)
        rtanh = tf.math.tanh(r)
        return inputs * tf.concat([rtanh, rtanh], axis=-1)

class PolarToCartesianLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PolarToCartesianLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.concat([tf.math.sin(inputs), tf.math.cos(inputs)], axis=-1)

class CartesianToPolar(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CartesianToPolar, self).__init__(**kwargs)

    def call(self, inputs):
        shape = (-1, 2, inputs.shape[-1]//2)
        tmp = tf.reshape(inputs, shape)
        tmp = tf.add(tmp, 1e-6*tf.random.normal((2, shape[-1])))
        x = tmp[:,0,:]
        y = tmp[:,1,:]
        theta = tf.atan2(-y, x) + np.pi/2
        return theta

class NormalizeCartesianRadius(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NormalizeCartesianRadius, self).__init__(**kwargs)

    def call(self, inputs):
        shape = (-1, 2, inputs.shape[-1]//2)
        tmp = tf.reshape(inputs, shape)
        tmp = tf.add(tmp, 1e-6*tf.random.normal((2, shape[-1])))
        x = tmp[:,0,:]
        y = tmp[:,1,:]
        theta = tf.atan2(y, x)
        x1 = tf.math.sin(theta)
        y1 = tf.math.cos(theta)
        return tf.concat([x1, y1], axis=-1)


class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=300.0, offset=0.0, normalize=True, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.scale = scale
        self.offset = offset
        self.normalize = normalize
        self.stretch = tf.Variable(scale, dtype=tf.float32, trainable=False)
        self.shift = tf.Variable(offset, dtype=tf.float32, trainable=False)

    def get_config(self):
        cfg = super().get_config()
        cfg['scale'] = self.scale
        cfg['offset'] = self.offset
        cfg['normalize'] = self.normalize
        return cfg

    def call(self, inputs):
        if self.normalize:
            i_std = tf.math.reduce_std(inputs, axis=-1, keepdims=True)
            i_mean = tf.math.reduce_mean(inputs, axis=-1, keepdims=True)
            i_centered = tf.subtract(inputs, i_mean)
            inputs = tf.divide(i_centered, i_std)
        i_scaled = tf.multiply(inputs, self.stretch)
        i_shifted = tf.add(i_scaled, self.shift)
        return i_shifted

def index_error(y_true, y_pred):
    y_true_norm = 2*np.pi/300 * y_true
    y_pred_norm = 2*np.pi/300 * y_pred
    x0, y0 = tf.sin(y_true_norm), tf.cos(y_true_norm)
    x1, y1 = tf.sin(y_pred_norm), tf.cos(y_pred_norm)
    sq_err = (x1-x0)**2 + (y1-y0)**2
    return tf.reduce_mean(sq_err, axis=-1)

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
        a = tf.math.floormod(a, self.n_pins)
        b = tf.math.floormod(b, self.n_pins)
        a = tf.cast(a, dtype=tf.int32)
        b = tf.cast(b, dtype=tf.int32)
        tmp1 = tf.gather(self.t_seg2idx, a, axis=0, batch_dims=1)
        tmp2 = tf.gather(tmp1, b, axis=2, batch_dims=1)
        idx = tf.linalg.diag_part(tmp2)
        return idx

    def idx_error(self, a_true, b_true, a_test, b_test):
        true_idx = self.t_path2mask_idx(a_true, b_true)
        true_mask = tf.math.bincount(true_idx, minlength=self.size, dtype=tf.int32, axis=-1)

        pred_idx = self.t_path2mask_idx(a_test, b_test)
        pred_mask = tf.math.bincount(pred_idx, minlength=self.size, dtype=tf.int32, axis=-1)
        err_mask = true_mask-pred_mask
        err = tf.cast(tf.square(err_mask), dtype=tf.float32) * self.t_lengths

        return err

    @tf.custom_gradient
    def loss_function(self, y_true, y_pred):
        y_true_pad = tf.pad(y_true, [[0,0],[1,0]])
        y_pred_pad = tf.pad(y_pred, [[0,0],[1,0]])

        a_true, b_true = (y_true_pad[:,:-1], y_true)
        a_pred, b_pred = (y_pred_pad[:,:-1], y_pred)

        err0 = self.idx_error(a_true, b_true, a_pred, b_pred)
        norm = tf.reduce_sum(0 * err0 + 1, axis=-1)
        loss = tf.divide(tf.reduce_sum(err0, axis=-1), norm)

        err1 = self.idx_error(a_true, b_true, a_pred, b_pred+1)
        pred_idx = self.t_path2mask_idx(a_pred, b_pred+1)
        t_grad = tf.gather((err1-err0), pred_idx, batch_dims=1)

        def gradient(y):
            y1 = tf.cast(y, dtype=tf.float32)
            g = tf.transpose(tf.transpose(t_grad) * y1)
            return (None, g)

        return loss, gradient

    # @tf.custom_gradient
    # def loss_function(self, y_true, y_pred):
    #     # Add a non-trainable implicit "0" pin to the
    #     # start of the results (i.e. assume all paths start at pin 0)
    #     y_true_pad = tf.pad(y_true, [[0,0],[1,0]])
    #     y_pred_pad = tf.pad(y_pred, [[0,0],[1,0]])

    #     a_true, b_true = (y_true_pad[:,:-1], y_true_pad[:,1:])
    #     a_pred, b_pred = (y_pred_pad[:,:-1], y_pred_pad[:,1:])

    #     true_idx = self.t_path2mask_idx(a_true, b_true)
    #     true_mask = tf.math.bincount(true_idx, minlength=self.size, dtype=tf.int32, axis=-1)

    #     pred_idx = self.t_path2mask_idx(a_pred, b_pred)
    #     pred_mask = tf.math.bincount(pred_idx, minlength=self.size, dtype=tf.int32, axis=-1)
    #     err_mask = true_mask-pred_mask
    #     err = tf.cast(tf.square(err_mask), dtype=tf.float32) #* self.t_lengths

    #     loss = tf.reduce_sum(err, axis=-1)
    #     norm = tf.reduce_sum(0 * err + 1, axis=-1)
    #     loss = tf.divide(loss, norm)

    #     pred_idx1 = self.t_path2mask_idx(a_pred, b_pred+1)
    #     pred_mask1 = tf.math.bincount(pred_idx1, minlength=self.size, dtype=tf.int32, axis=-1)
    #     err_mask1 = pred_mask1 - true_mask
    #     err1 = tf.cast(tf.square(err_mask1), dtype=tf.float32) #* self.t_lengths

    #     t_grad = tf.gather((err1-err), pred_idx1, batch_dims=1)

    #     def gradient(y):
    #         y1 = tf.cast(y, dtype=tf.float32)
    #         g = tf.transpose(tf.transpose(t_grad) * y1)
    #         return (None, g)

    #     return loss, gradient
