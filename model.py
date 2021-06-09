from typing import Any
import tensorflow as tf
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.module import module

class TangledModel(tf.keras.Model):
    def __init__(self, res:int=256, n_pins:int=256, name:str=None) -> None:
        self.res = res
        self.n_pins = n_pins

        inputs = tf.keras.Input(shape=(res,res,1), dtype='uint8')
        structure = [
            tf.keras.layers.experimental.preprocessing.Rescaling(-1./127.5, offset=1.0),
                # PeriodicPadding(1),
                # tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid'),
                # tf.keras.layers.BatchNormalization(),
                # tf.keras.layers.LeakyReLU(),
                # PeriodicPadding(1),
                # tf.keras.layers.Conv2D(64, 4, strides=2, padding='valid'),
                # tf.keras.layers.BatchNormalization(),
                # tf.keras.layers.LeakyReLU(),
                # PeriodicPadding(1),
                # tf.keras.layers.Conv2D(128, 4, strides=2, padding='valid'),
                # tf.keras.layers.LeakyReLU(),
                # PeriodicPadding(1),
                # tf.keras.layers.Conv2D(256, 4, strides=2, padding='valid'),
                # tf.keras.layers.LeakyReLU(),
                # PeriodicPadding(1),
                # tf.keras.layers.Conv2D(512, 4, strides=2, padding='valid'),
                # tf.keras.layers.LeakyReLU(),
                # PeriodicPadding(1),
                # tf.keras.layers.Conv2D(512, 4, strides=2, padding='valid'),
                # tf.keras.layers.LeakyReLU(),
                # tf.keras.layers.Dropout(0.5),
                # tf.keras.layers.Conv2DTranspose(512, 8, strides=2, padding='same'),
                # tf.keras.layers.LeakyReLU(),
                # tf.keras.layers.Dropout(0.5),
                # tf.keras.layers.Conv2DTranspose(512, 8, strides=2, padding='same'),
                # tf.keras.layers.LeakyReLU(),
                # tf.keras.layers.Conv2DTranspose(256, 8, strides=2, padding='same'),
                # tf.keras.layers.LeakyReLU(),
                # tf.keras.layers.Conv2DTranspose(128, 8, strides=2, padding='same'),
                # tf.keras.layers.LeakyReLU(),
                # tf.keras.layers.Conv2DTranspose(64, 8, strides=2, padding='same'),
                # tf.keras.layers.LeakyReLU(),
                # tf.keras.layers.Conv2DTranspose(32, 8, strides=2, padding='same'),
            Encoder.module,
            # tf.keras.layers.Activation(tf.keras.activations.swish),
            # tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same'),
            # tf.keras.layers.Cropping2D(cropping=((n_pins//2,n_pins//2),(n_pins//2,n_pins//2))),
            tf.keras.layers.Reshape((n_pins, n_pins)),
        ]
        outputs = self.structure_to_module(structure, inputs)
        super(TangledModel, self).__init__(inputs, outputs, name=name)

    @classmethod
    def structure_to_module(cls, structure:Any, inputs:tf.Module) -> tf.Module:
        if type(structure) is list:
            assert len(structure) > 0
            result = structure[0](inputs)
            for substructure in structure:
                result = cls.structure_to_module(substructure, result)
            return result
        else:
            return structure(inputs)

class PeriodicPadding(tf.keras.layers.Layer):
    def __init__(self, pad:int=1, name=None):
        super().__init__(name=name)
        self.pad_int = pad
        self.pad = tf.constant(pad, dtype=tf.int64)

    def get_config(self):
        config = super(PeriodicPadding, self).get_config()
        config.update({ "pad": self.pad_int })
        return config

    @tf.function
    def call(self, inputs):
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
    def downsample(filters, size, apply_batchnorm=True) -> tf.Module:
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
    def upsample(filters, size, apply_dropout=False) -> tf.Module:
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
            padding='same', kernel_initializer=initializer, use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        # result.add(tf.keras.layers.LeakyReLU()),
        result.add(tf.keras.layers.Activation(tf.keras.activations.swish))

        return result

    @classmethod
    def module(cls, inputs:tf.Module) -> tf.Module:
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

        encoder = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', # activation='sigmoid',
            kernel_initializer=tf.random_normal_initializer(0., 0.02))(encoder)

        return encoder
