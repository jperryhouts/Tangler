#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import cross_val_score

from typing import Tuple

from glob import glob
from PIL import Image, ImageOps

def pin_to_coord(pin:int, k:int=300) -> Tuple[float, float]:
    theta = pin * (2*np.pi/300)
    x = np.sin(theta)/2+0.5
    y = np.cos(theta)/2+0.5
    return np.array([x, y])

def load_img(img: str, resolution:Tuple[int,int]=(600,600)) -> np.array:
    img = Image.open(img) if type(img) is str else img
    img = ImageOps.grayscale(img)

    crop = None
    if img.size[0] != img.size[1]:
        cx, cy = (img.size[0]//2, img.size[1]//2)
        size2 = min(img.size)//2
        crop = (cx-size2, cy-size2, cx+size2, cy+size2)
    img = img.resize(resolution, box=crop)
    return np.array(img).astype(np.uint32)

def load_examples(N:int=None):
    images = glob("train/*/*.JPEG")
    for fname in images[:N]:
        pixels = load_img(fname).flatten()
        features = np.zeros(pixels.size+2, dtype=np.uint32)
        features[2:] = pixels

        target = np.zeros(300, dtype=np.float32)
        prev_pin = None
        for i, pin in enumerate(np.loadtxt(fname[:-4]+'tsv')):
            if prev_pin is not None:
                features[0] = i
                features[1] = prev_pin
                target *= 0
                target[int(pin)] = 1
                yield (features, target)

            prev_pin = pin

def main():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape=(600*600+2),
        batch_size=64))
    model.add(tf.keras.layers.Dense(300, activation='relu'))
    model.add(tf.keras.layers.Dense(300, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])
        #loss='mse')

    dataset = tf.data.Dataset.from_generator(
        load_examples,
        output_types=(tf.uint32, tf.float32),
        output_shapes=((600*600+2,), (300,)))
    batched_dataset = dataset.batch(64, drop_remainder=True)

    model.fit(batched_dataset, epochs=1)

    model.save("model")

if __name__ == '__main__':
    main()
