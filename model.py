from typing import Union, Any, Iterable
import logging
import numpy as np
import tensorflow as tf

import utils

TensorOrIterable = Union[tf.Tensor,Iterable[tf.Tensor]]

class TangledModel(tf.keras.Model):
    def __init__(self, n_pins:int=256, name:str=None) -> None:
        self.n_pins = n_pins

        inputs = tf.keras.layers.Input(shape=(256,256,1))
        preprocessed = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)(inputs)
        preprocessed = tf.keras.layers.experimental.preprocessing.RandomContrast(0.2)(preprocessed)

        # preprocessed = tf.keras.layers.Conv2D(3, 4, strides=1, padding='SAME')(preprocessed)
        # encoder = Encoder.mobilenet_encoder(preprocessed)
        # encoder = tf.keras.layers.Concatenate()([encoder, preprocessed])
        # encoder = tf.keras.layers.Conv2D(1, 4, strides=1, padding='SAME')(encoder)
        # outputs = tf.keras.layers.Reshape((n_pins, n_pins))(encoder)

        # downsampled = Encoder.downsample(3, 4, apply_batchnorm=False)(inputs) # 256x256 -> 128x128
        # encoder = Encoder.mobilenet_encoder(downsampled)
        # encoder = tf.keras.layers.Concatenate()([encoder, downsampled])
        # encoder = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='SAME',
        #     kernel_initializer=tf.random_normal_initializer(0., 0.02),
        #     kernel_regularizer=tf.keras.regularizers.l2())(encoder) # 128x128 -> 256x256
        # outputs = tf.keras.layers.Reshape((n_pins, n_pins))(encoder)

        encoder = Encoder.u_net(preprocessed)
        outputs = tf.keras.layers.Reshape((n_pins, n_pins))(encoder)

        # preproc = tf.keras.layers.Conv2D(3, 2, strides=1, padding='SAME', use_bias=False)(inputs)
        # encoder = Encoder.resnet50v2_encoder(preproc)
        # outputs = tf.keras.layers.Reshape((n_pins, n_pins))(encoder)

        # ds1 = Encoder.downsample(3, 4, apply_batchnorm=False)(inputs) # 256x256 -> 128x128
        # enc1 = Encoder.mobilenet_encoder(ds1)
        # ds2 = Encoder.downsample(3, 4, apply_batchnorm=False)(inputs) # 256x256 -> 128x128
        # enc2 = Encoder.mobilenet_encoder(ds2)
        # ds3 = Encoder.downsample(3, 4, apply_batchnorm=False)(inputs) # 256x256 -> 128x128
        # enc3 = Encoder.mobilenet_encoder(ds3)
        # ds4 = Encoder.downsample(3, 4, apply_batchnorm=False)(inputs) # 256x256 -> 128x128
        # enc4 = Encoder.mobilenet_encoder(ds4)
        # encoders = tf.keras.layers.Concatenate()([enc1, ds1, enc2, ds2, enc3, ds3, enc4, ds4])
        # encoder = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='SAME',
        #     kernel_initializer=tf.random_normal_initializer(0., 0.02),
        #     kernel_regularizer=tf.keras.regularizers.l2())(encoders) # 128x128 -> 256x256
        # outputs = tf.keras.layers.Reshape((n_pins, n_pins))(encoder)

        # encoder = tf.keras.layers.Reshape((256, 256))(encoder)
        # outputs = FFTLayer(name='fft')(encoder)

        super(TangledModel, self).__init__(inputs=inputs, outputs=outputs, name=name)

    def img_to_tangle(self, image:np.ndarray) -> np.ndarray:
        image = image.reshape((1,256,256,1)).astype(np.float32)
        predicted = super(TangledModel, self).predict(image)
        tangle = utils.target_to_tangle(predicted)
        # tangle = (tangle + tangle.T)/2.0
        return tangle

class FFTLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs:TensorOrIterable) -> TensorOrIterable:
        x = tf.cast(inputs, tf.complex64)
        x_fft = tf.signal.fft2d(x)
        x_fft = tf.signal.fftshift(x_fft)
        x_fft_real = tf.math.real(x_fft)
        x_fft_imag = tf.math.imag(x_fft)
        result = tf.stack([x_fft_real, x_fft_imag], axis=-1)
        result = tf.cast(result, tf.float32)
        return result

class PeriodicPadding(tf.keras.layers.Layer):
    def __init__(self, pad:int=1, **kwargs) -> None:
        super(PeriodicPadding, self).__init__(**kwargs)
        self.pad = pad

    def get_config(self) -> dict[str,Any]:
        config = super(PeriodicPadding, self).get_config()
        config.update({ "pad": self.pad })
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
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2,
            # kernel_regularizer=tf.keras.regularizers.l2(),
            kernel_initializer=initializer, use_bias=False, padding='SAME'))

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
            kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
            kernel_initializer=initializer, use_bias=False, padding='SAME'))

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
        base_model = tf.keras.applications.MobileNetV2(input_shape=(256,256,3),
                                                        include_top=False,
                                                        weights='imagenet')

        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        down_stack = tf.keras.Model(base_model.input, base_model_outputs)

        down_stack.trainable = False

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

    @classmethod
    def resnet50v2_encoder(cls, inputs):
        base_model = tf.keras.applications.ResNet50V2(input_shape=(256,256,3),
                                                        include_top=False,
                                                        weights='imagenet')

        layer_names = [
            'conv2_block1_2_relu',   # 64x64
            'conv3_block3_2_relu',   # 32x32
            'conv4_block5_2_relu',   # 16x16
            'conv4_block6_2_relu',  # 8x8
            # 'block_16_project',      # 4x4
        ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        down_stack = tf.keras.Model(base_model.input, base_model_outputs)

        # down_stack.trainable = False

        up_stack = [
            # cls.upsample(512, 4),  # 4x4 -> 8x8
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
        decoder = cls.upsample(1,4)(decoder) #128x128 -> 256x256

        return decoder