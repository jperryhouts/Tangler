import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import Masks, load_img

def do_predict(paths:str, model_path:str, save_to:str=None, res:int=600, n_pins:int=300) -> None:
    masks = Masks(n_pins)
    model = tf.keras.models.load_model(model_path, custom_objects={'loss_function': masks.loss_function})

    images = []
    for img_path in paths:
        pixels = load_img(img_path, res).astype(np.float32)
        images.append(pixels.reshape((res,res,1)))

    result = model.predict(np.array(images))
    for i, pattern in enumerate(result):
        pins = pattern.astype(int)%n_pins
        print(pattern[:10], pins[:10])

        coords = [masks.pin2coord(pin) for pin in pins]
        x, y = np.array(coords).T
        _, ax = plt.subplots(2, 2)
        ax[0][0].hist(pins, bins=range(0,n_pins))
        ax[0][0].set_xlim((0,n_pins))
        ax[1][0].plot(x, y, 'k-', lw=0.03)
        ax[1][0].set_aspect(1.0)

        pins = np.loadtxt(paths[i][:-5]+'.tsv').astype(int)[:pattern.size]
        coords = [masks.pin2coord(pin) for pin in pins]
        x, y = np.array(coords).T
        ax[0][1].hist(pins, bins=range(0,n_pins))
        ax[0][1].set_xlim((0,n_pins))
        ax[1][1].plot(x, y, 'k-', lw=0.03)
        ax[1][1].set_aspect(1.0)

    plt.show()
