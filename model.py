from typing import Union, Any, Iterable
import logging
import numpy as np
import tensorflow as tf

TensorOrIterable = Union[tf.Tensor,Iterable[tf.Tensor]]

class TangledModel(tf.keras.Model):
    def __init__(self, n_pins:int=256, name:str=None) -> None:
        self.n_pins = n_pins

        inputs = tf.keras.layers.Input(shape=(256,256,1))

        downsampled = Encoder.downsample(3, 4, apply_batchnorm=False)(inputs) # 256x256 -> 128x128
        decoder = Encoder.mobilenet_encoder(downsampled)
        decoder = tf.keras.layers.Concatenate()([decoder, downsampled])
        decoder = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='SAME',
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            kernel_regularizer=tf.keras.regularizers.l2())(decoder) # 128x128 -> 256x256

        outputs = tf.keras.layers.Reshape((n_pins, n_pins))(decoder)
        super(TangledModel, self).__init__(inputs=inputs, outputs=outputs, name=name)

    def tangle_img(self, inputs:np.ndarray) -> np.ndarray:
        img = inputs.reshape((1,256,256,1))
        prediction = super(TangledModel, self).predict(img.astype(np.float32))

        prediction = prediction.reshape((self.n_pins,self.n_pins))
        pred_sym = (prediction + prediction.T)/2.0
        return pred_sym

    @staticmethod
    def untangle(prediction:np.ndarray, path_len:int,
            threshold:Union[float,str]='60%', dtype:np.dtype=np.uint8) -> np.ndarray:

        if type(threshold) is str:
            percentile = float(threshold[:-1])
            threshold = np.percentile(prediction, percentile)
            # print("Threshold: %g"%threshold)

        pmin = prediction.min()
        pred_l = np.tril(prediction-pmin, k=-2) + pmin
        locs = np.where(pred_l > threshold)

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

class PeriodicPadding(tf.keras.layers.Layer):
    def __init__(self, pad:int=1, name:str=None) -> None:
        super().__init__(name=name)
        self.pad_int = pad
        self.pad = tf.constant(pad, dtype=tf.int64)

    def get_config(self) -> dict[str,Any]:
        config = super(PeriodicPadding, self).get_config()
        super(PeriodicPadding, self).call()
        config.update({ "pad": self.pad_int })
        return config

    @tf.function
    def call(self, inputs:TensorOrIterable) -> TensorOrIterable:
        pad = self.pad
        upper_pad = inputs[:,-pad:,:]
        lower_pad = inputs[:,:pad,:]

        partial = tf.concat([upper_pad, inputs, lower_pad], axis=1)

        left_pad = partial[:,:,-pad:]
        right_pad = partial[:,:,:pad]

        padded = tf.concat([left_pad, partial, right_pad], axis=2)

        return padded

class Encoder():
    '''
    This class is only meant to be a container for a few static methods.
    In particular, the `module` method, which returns a tf.Module object
    representing a complete (auto)encoder.

    This class contains a lot of code borrowed from
    the TensorFlow Pix2Pix tutorial:
    https://www.tensorflow.org/tutorials/generative/pix2pix
    '''

    @staticmethod
    def downsample(filters, size:int, apply_batchnorm:bool=True) -> tf.Module:
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        # result.add(tf.keras.layers.LeakyReLU())
        result.add(tf.keras.layers.Activation(tf.keras.activations.swish))

        return result

    @staticmethod
    def upsample(filters, size:int, apply_dropout:bool=False) -> tf.Module:
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
            kernel_regularizer=tf.keras.regularizers.l2(), padding='SAME',
            kernel_initializer=initializer, use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        # result.add(tf.keras.layers.LeakyReLU()),
        result.add(tf.keras.layers.Activation(tf.keras.activations.swish))

        return result

    @classmethod
    def u_net(cls, inputs:tf.Module) -> tf.Module:
        down_stack = [
            cls.downsample(64,  4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            cls.downsample(128, 4),  # (bs, 64, 64, 128)
            cls.downsample(256, 4),  # (bs, 32, 32, 256)
            cls.downsample(512, 4),  # (bs, 16, 16, 512)
            cls.downsample(512, 4),  # (bs, 8, 8, 512)
            cls.downsample(512, 4),  # (bs, 4, 4, 512)
            cls.downsample(512, 4),  # (bs, 2, 2, 512)
            cls.downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            cls.upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            cls.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            cls.upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            cls.upsample(512, 4),  # (bs, 16, 16, 1024)
            cls.upsample(256, 4),  # (bs, 32, 32, 512)
            cls.upsample(128, 4),  # (bs, 64, 64, 256)
            cls.upsample(64,  4),  # (bs, 128, 128, 128),
        ]

        encoder = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            encoder = down(encoder)
            skips.append(encoder)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            encoder = up(encoder)
            encoder = tf.keras.layers.Concatenate()([encoder, skip])

        encoder = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same',
            kernel_initializer=tf.random_normal_initializer(0., 0.02))(encoder)

        return encoder

    @classmethod
    def mobilenet_encoder(cls, inputs):
        base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(128,128,3))

        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        down_stack = tf.keras.Model(base_model.input, base_model_outputs)

        # down_stack.trainable = False

        up_stack = [
            cls.upsample(512, 4),  # 4x4 -> 8x8
            cls.upsample(512, 4),  # 8x8 -> 16x16
            cls.upsample(256, 4),  # 16x16 -> 32x32
            cls.upsample(128, 4),   # 32x32 -> 64x64
        ]

        # Downsampling through the model
        skips = down_stack(inputs)
        decoder = skips[-1]
        skips = list(reversed(skips[:-1]))

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            decoder = up(decoder)
            decoder = tf.keras.layers.Concatenate()([decoder, skip])

        # This is the last layer of the model
        decoder = cls.upsample(64,4)(decoder) #64x64 -> 128x128

        return decoder
