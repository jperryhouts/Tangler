#!/usr/bin/env python3

import itertools
import time
from typing import Any
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from glumpy import app, gloo, gl
import utils

class TangledModel(tf.keras.Model):
    def __init__(self, model_path:str, path_len:int=8000) -> None:
        self.path_len = path_len
        base_model = tf.keras.models.load_model(model_path)
        super().__init__(base_model.input, base_model.layers[-4].output)
        self.res = self.input.type_spec.shape[1]
        self.n_pins = self.output.type_spec.shape[2]

    def theta_to_pins(self, thetas:np.ndarray) -> np.ndarray:
        ppins = (self.n_pins*thetas/(2*np.pi)).astype(np.int)%self.n_pins
        ppins = ppins.tolist()

        pins = [0]
        while len(pins) < self.path_len:
            prev_pin = pins[-1]
            options = ppins[prev_pin]
            next_pin = options.pop(0) if len(options) > 0 else (prev_pin+1)%self.n_pins
            if next_pin == prev_pin:
                next_pin = options.pop(0) if len(options) > 0 else (prev_pin+1)%self.n_pins
            pins.append(next_pin)

        return np.array(pins)

    def predict(self, inputs:np.ndarray) -> np.ndarray:
        img = inputs.astype(np.float32).reshape((1,self.res,self.res,1))
        thetas = super().predict(img)[0][0]
        path = self.theta_to_pins(thetas)
        return path

class Camera():
    def __init__(self, model_res:int, capture_source:Any=0):
        self.camera = cv2.VideoCapture(capture_source)
        self.load_crop_dimensions()
        self.model_res = model_res

    def close(self):
        print('Releasing camera')
        self.camera.release()

    def load_crop_dimensions(self):
        w, h = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == h:
            self.y0, self.x0 = [0,0]
            self.y1, self.x1 = [h-1,w-1]
        else:
            imres = min(w,h)
            wc, hc = w//2, h//2
            self.y0, self.x0 = (hc-imres//2, wc-imres//2)
            self.y1, self.x1 = (hc+imres//2, wc+imres//2)

    def img_stream(self):
        while self.camera.isOpened():
            success, frame = self.camera.read()

            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = img[self.y0:self.y1,self.x0:self.x1]
                img = cv2.resize(img, (self.model_res, self.model_res))
                yield img

class ImageIterator():
    def __init__(self, source:Any, infinite:bool, res:int) -> None:
        if source == 'webcam':
            self.type = 'webcam'
            self.cam = Camera(res)
            self.source = self.cam.img_stream()
        else:
            self.type = 'files'
            self.source = itertools.cycle(source) if infinite else itertools.chain(source)

    def close(self) -> None:
        print("Closing ImageIterator")
        if self.type == 'webcam':
            self.cam.close()

    def __next__(self):
        return self.source.__next__()


class Visualizer():
    def __init__(self, model:tf.keras.Model, source:ImageIterator, backend:str='opengl'):
        self.model = model
        self.mapping = utils.Mapping(self.model.n_pins)
        self.source = source

        self.backend = backend
        if backend == 'matplotlib':
            plt.ion()
            _, self.axes = plt.subplots(1, 2)

    def close(self):
        print("Closing Visualizer")
        self.source.close()
        if self.backend == 'matplotlib':
            plt.ioff()

    def run(self):
        if self.backend == 'matplotlib':
            self.axes[1].set_xlim((-1,1))
            self.axes[1].set_ylim((-1,1))
            self.axes[1].set_aspect(1.0)

            for img in self.source:
                self.axes[0].clear()
                self.axes[0].imshow(img.reshape((self.model.res, self.model.res)), cmap=plt.cm.gray, vmin=0, vmax=255)

                path = self.model.predict(img)
                x, y = self.mapping.pins2xy(path)
                self.axes[1].clear()
                self.axes[1].plot(x, y, 'k-', lw=0.01)

                plt.draw()
                plt.pause(0.1)
        else:
            window = app.Window(512, 512, color=(1,1,1,1))
            prog = gloo.Program(count=self.model.path_len,
                vertex = '''
                    attribute float pin;
                    void main() {
                        float PI = 3.1415926535897932384626433832795;
                        float n_pins = ''' + str(self.model.n_pins) + ''';
                        float theta = 2.0 * PI * pin / n_pins;
                        float x = sin(theta), y = cos(theta);
                        gl_Position = vec4(x, y, 0.0, 1.0);
                    }
                    ''',
                fragment = '''
                    void main() {
                        gl_FragColor = vec4(0.0, 0.0, 0.0, 1e-1);
                    }
                    ''')
            prog["pin"] = np.zeros((self.model.path_len,), dtype=np.float32)

            @window.event
            def on_draw(dt):
                img = self.source.__next__()
                path = self.model.predict(img)

                window.clear()
                prog.draw(gl.GL_LINE_STRIP)
                prog["pin"][:] = path.astype(np.float32)

                if self.source.type == "files":
                    time.sleep(0.5)

            app.run()

        self.close()

def do_demo(model_path:str, source:Any, backend:str, infinite:bool):
    model = TangledModel(model_path)
    images = ImageIterator(source, infinite, model.res)
    vis = Visualizer(model, images, backend)
    vis.run()
