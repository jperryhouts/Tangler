
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

import utils

class Mapping():
    def __init__(self, n_pins:int) -> None:
        self.n_pins = n_pins
        self.pin_coords = self.get_pin_mapping(n_pins)

    def get_pin_mapping(self, n_pins:int) -> np.ndarray:
        thetas = np.arange(n_pins) * 2 * np.pi / n_pins
        coords = np.zeros((n_pins, 2))
        coords[:,0] = np.sin(thetas)/2+1
        coords[:,1] = np.cos(thetas)/2+1
        return coords

    def pins2xy(self, pins:np.ndarray) -> np.ndarray:
        return self.pin_coords[pins].T

def plot_path(pins:np.ndarray, mapping:Mapping, ax:plt.axes) -> None:
    true_x, true_y = mapping.pins2xy(pins)
    ax.plot(true_x, true_y, 'k-', lw=0.01)
    ax.set_aspect(1.0)

def plot_pdf(masks:np.ndarray, mapping:Mapping, ax:plt.axes) -> None:
    n_pins = mapping.n_pins
    for i in range(n_pins):
        for j in range(i+1,n_pins):
            if masks[i*n_pins+j] > 0.01:
                x, y = mapping.pins2xy(np.array([[i, j]]))
                ax.plot(x, y, 'k-', lw=0.01)
                ax.set_aspect(1.0)

def do_predict(paths:str, model_path:str, res:int=600, n_pins:int=300) -> None:
    loss_function = utils.d_theta_loss(n_pins)
    model = tf.keras.models.load_model(model_path, custom_objects={
        'loss_function': loss_function,
        'index_error': loss_function,
        'dtheta': utils.d_theta_loss(n_pins),
        'convolution_loss': utils.convolution_loss,
        'mask_comparison_loss': utils.get_mask_comparison_loss(n_pins)
        })

    images = []
    for img_path in paths:
        pixels = utils.load_img(img_path, res).astype(np.float32)
        images.append(pixels.reshape((res,res,1)))

    X,Y = np.mgrid[-1:1:1j*res, -1:1:1j*res]
    R = np.sqrt(X**2 + Y**2)
    circle_border = np.where(R>1)

    mapping = Mapping(n_pins)
    results = model.predict(np.array(images))

    for i, pattern in enumerate(results):
        _, ax = plt.subplots(2, 2)

        im = images[i].reshape((res,res))
        im[circle_border] = 255
        ax[0][0].imshow(im, cmap=plt.cm.gray, vmin=0, vmax=255)

        pins = np.loadtxt(paths[i][:-5]+'.tsv').astype(int) #[:pattern.size]
        sns.histplot(pins, bins=range(0,n_pins), color='blue', label='true', kde=True, ax=ax[0][1])
        plot_path(pins, mapping, ax[1][0])

        #pins = pattern.astype(int)%n_pins
        #sns.histplot(pins, bins=range(0,n_pins), color='orange', label='predicted', kde=True, ax=ax[0][1])
        plot_pdf(pattern, mapping, ax[1][1])

        ax[0][1].set_xlim((0,n_pins))
        ax[0][1].legend()

    print(model.summary())
    print("Model:",model_path)
    for i, pattern in enumerate(results):
        for t in (0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            print(t, np.where(pattern > t)[0].size)
        # plt.figure()
        # plt.hist(pattern[np.where(pattern > 0)], bins=150)
        print(np.where(pattern > 0)[0].shape)
        print('%g/%g __ %g+/-%g'%(pattern.min(), pattern.max(), pattern.mean(), pattern.std()))
        print('   ', pattern[:4], (pattern.astype(int)%n_pins)[:4])

    plt.show()
