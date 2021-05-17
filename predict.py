import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import Masks, load_img

def do_predict(fname:str, model_path:str, save_to:str=None, res:int=600, n_pins:int=300) -> None:
    masks = Masks(n_pins)

    model = tf.keras.models.load_model(model_path, custom_objects={'loss_function': masks.loss_function})

    pixels = load_img(fname, res).astype(np.float32)
    pixels = pixels.flatten().reshape((1,res,res,1))
    result = model.predict(pixels)[0]
    if save_to:
        np.savetxt(save_to, result)

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

