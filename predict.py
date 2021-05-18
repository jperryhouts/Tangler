
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import utils
from utils import Masks, load_img, ScaleLayer, lambda_activation

def do_predict(paths:str, model_path:str, save_to:str=None, res:int=600, n_pins:int=300) -> None:
    masks = Masks(n_pins)
    model = tf.keras.models.load_model(model_path, custom_objects={
        'loss_function': masks.loss_function,
        'ScaleLayer': ScaleLayer,
        'lambda_activation': lambda_activation,
        'index_error': utils.index_error
        })

    images = []
    for img_path in paths:
        pixels = load_img(img_path, res).astype(np.float32)
        images.append(pixels.reshape((res,res,1)))

    X,Y = np.mgrid[-1:1:1j*res, -1:1:1j*res]
    R = np.sqrt(X**2 + Y**2)
    circle_border = np.where(R>1)

    result = model.predict(np.array(images))

    for i, pattern in enumerate(result):
        _, ax = plt.subplots(2, 2)

        im = images[i].reshape((res,res))
        im[circle_border] = 255
        ax[0][0].imshow(im, cmap=plt.cm.gray, vmin=0, vmax=255)

        true_pins = np.loadtxt(paths[i][:-5]+'.tsv').astype(int)[:pattern.size]
        true_coords = [masks.pin2coord(pin) for pin in true_pins]
        true_x, true_y = np.array(true_coords).T
        sns.distplot(true_pins, bins=range(0,n_pins), color='blue', label='true', ax=ax[0][1])
        #ax[0][1].hist(true_pins, bins=range(0,n_pins), color='blue', label='true')
        ax[1][1].plot(true_x, true_y, 'k-', lw=0.03)
        ax[1][1].set_aspect(1.0)

        pins = pattern.astype(int)%n_pins
        coords = [masks.pin2coord(pin) for pin in pins]
        x, y = np.array(coords).T
        sns.distplot(pins, bins=range(0,n_pins), color='orange', label='predicted', ax=ax[0][1])
        #ax[0][1].hist(pins, bins=range(0,n_pins), color='orange', alpha=0.8, label='predicted')

        ax[1][0].plot(x, y, 'k-', lw=0.03)
        ax[1][0].set_aspect(1.0)

        ax[0][1].set_xlim((0,n_pins))
        ax[0][1].set_ylim((0,0.015))
        ax[0][1].legend()

    print(model.summary())
    print("Model:",model_path)
    for i, pattern in enumerate(result):
        pins = pattern.astype(int)%n_pins
        print(pattern.min(), pattern.max(), pattern.mean(), pattern.std(), pattern[:4], pins[:4])

    plt.show()
