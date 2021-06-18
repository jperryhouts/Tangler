from typing import Optional
import tensorflow as tf
# import tensorflow_model_optimization as tfmot

IMG_RES = 256
N_PINS = 256

def downsample(filters:int, size:int, apply_batchnorm:bool=True) -> tf.Module:
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2,
        # kernel_regularizer=tf.keras.regularizers.l2(),
        kernel_initializer=initializer, use_bias=False, padding='SAME'))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    # result.add(tf.keras.layers.Activation(tf.keras.activations.swish))

    return result

def upsample(filters:int, size:int, apply_dropout:bool=False) -> tf.Module:
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
        kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4),
        kernel_initializer=initializer, use_bias=False, padding='SAME'))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.LeakyReLU())
    # result.add(tf.keras.layers.Activation(tf.keras.activations.swish))

    return result

def encoder_stack():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(128,128,3),
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

    return tf.keras.Model(base_model.input, base_model_outputs, name="down_stack")

def SimpleModel(encoder:Optional[tf.keras.Model]=None, name:str="Tangler") -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(IMG_RES,IMG_RES,1), dtype=tf.float32)
    downsampled = downsample(3, 4, apply_batchnorm=False)(inputs) # 256x256x1 -> 128x128x3

    ###
    ### Begin "auto-encoder"
    ###
    if encoder is None:
        encoder = encoder_stack()

    decoder_stack = [
        upsample(512, 4),  # 4x4 -> 8x8
        upsample(512, 4),  # 8x8 -> 16x16
        upsample(256, 4),  # 16x16 -> 32x32
        upsample(128, 4),   # 32x32 -> 64x64
    ]

    # Downsampling through the model
    skips = encoder(downsampled)
    layers = skips[-1]
    skips = list(reversed(skips[:-1]))

    # Upsampling and establishing the skip connections
    for up, skip in zip(decoder_stack, skips):
        print(type(skip), skip)
        layers = up(layers)
        layers = tf.keras.layers.Concatenate()([layers, skip])

    layers = upsample(64,4)(layers) #64x64 -> 128x128
    ###
    ### End "auto-encoder"
    ###

    # One final upsampling / skip connection to match target shape
    layers = tf.keras.layers.Concatenate()([layers, downsampled])
    layers = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='SAME',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        kernel_regularizer=tf.keras.regularizers.l2()
    )(layers) # 128x128 -> 256x256
    outputs = tf.keras.layers.Reshape((N_PINS, N_PINS))(layers)

    return tf.keras.Model(inputs, outputs, name=name)
