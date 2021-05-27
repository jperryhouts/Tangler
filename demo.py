#!/usr/bin/env python3

import time
from typing import Any
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glumpy import app, gloo, gl
import utils

from utils import TangledModel, ImageIterator

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

                path = self.model.predict_convert(img)
                x, y = self.mapping.pins2xy(path)
                self.axes[1].clear()
                self.axes[1].plot(x, y, 'k-', lw=0.01)

                plt.draw()
                plt.pause(0.1)
        else:
            window = app.Window(512, 512, color=(1,1,1,1))
            prog = gloo.Program(count=self.model.max_path_len,
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
            prog["pin"] = np.zeros((self.model.max_path_len,), dtype=np.float32)

            @window.event
            def on_draw(dt):
                img = self.source.__next__()
                path = self.model.predict_convert(img)
                window.clear()
                prog.draw(gl.GL_LINE_STRIP)
                prog["pin"][:] = np.zeros((self.model.max_path_len,), dtype=np.float32)
                prog["pin"][:path.size] = path.astype(np.float32)

                if self.source.type == "files":
                    time.sleep(0.5)

            app.run()

        self.close()

def do_demo(model_path:str, source:Any, backend:str, cycle:bool):
    model = TangledModel(model_path)
    images = ImageIterator(source, cycle, model.res)
    vis = Visualizer(model, images, backend)
    vis.run()
