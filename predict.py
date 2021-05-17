#!/usr/bin/env python3

from .utils import Masks, load_img

from argparse import ArgumentParser

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
#tf.config.set_visible_devices([], 'GPU')
tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--resolution', '-r', type=int, default=600)
    parser.add_argument('--pins', '-k', type=int, default=300)
    parser.add_argument('--model', '-m', help='path to tfmodel', required=True)
    parser.add_argument('fname', help='Image to convert into thread pattern')
    args = parser.parse_args()

    masks = Masks(args.pins)

    model = tf.keras.models.load_model(args.model, custom_objects={'loss_function': masks.loss_function})

    fname = args.fname
    pixels = load_img(fname, args.resolution).astype(np.float32)
    pixels = pixels.flatten().reshape((1,args.resolution,args.resolution,1))
    result = model.predict(pixels)[0]#.round().astype(int)
    prediction = (args.fname[:-4]+'pred')
    np.savetxt(prediction, result)

    for idx, count in enumerate((1*(result > 0.1)).astype(int)):
        if count:
        #for _ in range(count):
            a, b = np.array(masks.idx2pair(idx))
            ac = masks.pin2coord(a)
            bc = masks.pin2coord(b)
            x = [ac[0], bc[0]]
            y = [ac[1], bc[1]]
            #sys.stdout.write(f'{x}, {y} -- ')
            plt.plot(x, y, 'k-', lw=0.1)

    plt.show()