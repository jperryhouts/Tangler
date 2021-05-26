
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import utils

def plot_path(pins:np.ndarray, mapping:utils.Mapping, ax:plt.axes) -> None:
    true_x, true_y = mapping.pins2xy(pins)
    ax.plot(true_x, true_y, 'k-', lw=0.01)
    ax.set_aspect(1.0)

def do_predict(paths:str, model_path:str, res:int=600, n_pins:int=300) -> None:
    base_model = tf.keras.models.load_model(model_path, custom_objects={})
    # base_model = tf.train.load_checkpoint(model_path)
    model = tf.keras.Model(base_model.input, base_model.layers[-4].output)

    images = []
    for img_path in paths:
        pixels = utils.load_img(img_path, res).astype(np.float32)
        images.append(pixels.reshape((res,res,1)))

    X,Y = np.mgrid[-1:1:1j*res, -1:1:1j*res]
    R = np.sqrt(X**2 + Y**2)
    circle_border = np.where(R>1)

    mapping = utils.Mapping(n_pins)

    results = model.predict(np.array(images))
    #print(results.shape)
    #plt.hist(results.flatten(), bins=256)

    for i, ptheta in enumerate(results):
        ppins = (n_pins*ptheta[0]/(2*np.pi)).astype(np.int)%n_pins
        ppins = ppins.tolist()

        _, ax = plt.subplots(1, 2)

        im = images[i].reshape((res,res))
        im[circle_border] = 255
        ax[0].imshow(im, cmap=plt.cm.gray, vmin=0, vmax=255)

        pins = [0]
        while len(pins) < 10000:
            prev_pin = pins[-1]
            options = ppins[prev_pin]
            next_pin = options.pop(0) if len(options) > 0 else (prev_pin+1)%n_pins
            if next_pin == prev_pin:
                next_pin = options.pop(0) if len(options) > 0 else (prev_pin+1)%n_pins
            pins.append(next_pin)
        plot_path(pins, mapping, ax[1])

        # for pin_a, pairs in enumerate(ppins):
        #     for pin_b in pairs:
        #         if pin_b > pin_a:
        #             x, y = mapping.pins2xy(np.array([[pin_a, pin_b]]))
        #             ax[2].plot(x, y, 'k-', lw=0.01)
        # ax[2].set_aspect(1.0)

    #     pins = np.loadtxt(paths[i][:-5]+'.tsv').astype(int) #[:pattern.size]
    #     sns.histplot(pins, bins=range(0,n_pins), color='blue', label='true', kde=True, ax=ax[0][1])

    #     #pins = pattern.astype(int)%n_pins
    #     #sns.histplot(pins, bins=range(0,n_pins), color='orange', label='predicted', kde=True, ax=ax[0][1])
    #     plot_pdf(pattern, mapping, ax[1][1])

    #     ax[0][1].set_xlim((0,n_pins))
    #     ax[0][1].legend()

    # print(model.summary())
    # print("Model:",model_path)
    # for i, pattern in enumerate(results):
    #     for t in (0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
    #         print(t, np.where(pattern > t)[0].size)
    #     # plt.figure()
    #     # plt.hist(pattern[np.where(pattern > 0)], bins=150)
    #     print(np.where(pattern > 0)[0].shape)
    #     print('%g/%g __ %g+/-%g'%(pattern.min(), pattern.max(), pattern.mean(), pattern.std()))
    #     print('   ', pattern[:4], (pattern.astype(int)%n_pins)[:4])

    plt.show()
