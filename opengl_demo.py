import time
from typing import Optional, Iterable, Generator, Union
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from utils import load_img
from model import TangledModel

class Camera():
    def __init__(self, res:int, capture_source:int=0) -> None:
        self.camera = cv2.VideoCapture(capture_source)
        self.load_crop_dimensions()
        self.res = res
        self.load_cmask()

    def close(self) -> None:
        print('Releasing camera')
        self.camera.release()

    def load_crop_dimensions(self) -> None:
        w, h = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == h:
            self.y0, self.x0 = [0,0]
            self.y1, self.x1 = [h-1,w-1]
        else:
            imres = min(w,h)
            wc, hc = w//2, h//2
            self.y0, self.x0 = (hc-imres//2, wc-imres//2)
            self.y1, self.x1 = (hc+imres//2, wc+imres//2)

    def load_cmask(self) -> None:
        res = self.res
        X,Y = np.mgrid[-1:1:res*1j,-1:1:res*1j]
        R2 = X**2+Y**2
        self.cmask = (1*(R2<1.0)).astype(np.uint8)
        self.cmask_inv = 1-self.cmask

    def img_stream(self) -> Generator[np.ndarray, None, None]:
        while self.camera.isOpened():
            success, frame = self.camera.read()

            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = img[self.y0:self.y1,self.x0:self.x1]
                img = cv2.resize(img, (self.res, self.res))
                img = img/127.0-1.0
                img *= self.cmask
                yield img

class ImageIterator():
    def __init__(self, source:Union[str,Iterable], cycle:bool, res:int) -> None:
        if source == 'webcam':
            self.type = 'webcam'
            self.cam = Camera(res)
            self.source = self.cam.img_stream()
        else:
            self.type = 'files'
            self.paths = itertools.cycle(source) if cycle else itertools.chain(source)
            def gen():
                for path in self.paths:
                    yield load_img(path, res)
            self.source = gen()

    def close(self) -> None:
        print("Closing ImageIterator")
        if self.type == 'webcam':
            self.cam.close()

    def __next__(self):
        return self.source.__next__()

def show_prediction(pred, threshold:Optional[float]=None):
    _, ax = plt.subplots(1,1)
    # vrange = max(abs(pred.max()), abs(pred.min()))
    norm=None
    if pred.min() < 0 and pred.max() > 0:
        norm = colors.TwoSlopeNorm(vmin=pred.min(), vcenter=0, vmax = pred.max())
    im = ax.imshow(pred, aspect='auto', cmap=plt.cm.seismic, interpolation='nearest', norm=norm)
    if threshold is not None:
        ax.contour(pred, levels=[threshold], colors='k', linewidths=0.5)
    plt.colorbar(im, ax=ax)
    plt.show()

def do_demo(model_path:str, data_source, mirror:bool=False, cycle:bool=True,
        delay:int=0, path_len:int=60000, threshold:Union[str,float]='60%') -> None:

    model = TangledModel()
    model.load_weights(model_path)

    image_source = ImageIterator(data_source, cycle, 256)

    vertex_src = f'''
    in float pin;
    void main() {{
        float PI = 3.1415926535897932384626433832795;
        float n_pins = {model.n_pins};
        float theta = 2.0 * PI * pin.x / n_pins;
        float x = sin(theta), y = cos(theta);
        gl_Position = vec4(({'-x' if mirror else 'x'}), -y, 0.0, 1.0);
    }}
    '''

    fragment_src = '''
    void main() {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.04);
    }
    '''

    if not glfw.init():
        raise Exception("Cannot initialize glfw")

    window = glfw.create_window(512, 512, "Tangler Demo", None, None)

    if not window:
        glfw.terminate()
        raise Exception("glfw failed to open window")

    glfw.make_context_current(window)

    shader = compileProgram(
        compileShader(vertex_src, GL_VERTEX_SHADER),
        compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )

    thetas = np.zeros(path_len, dtype=np.float32)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, thetas.nbytes, thetas, GL_DYNAMIC_DRAW)

    buffer_loc = glGetAttribLocation(shader, "pin")
    glEnableVertexAttribArray(buffer_loc)
    glVertexAttribPointer(buffer_loc, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    glUseProgram(shader)
    glClearColor(1., 1., 1., 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    frame_count = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()

        img = image_source.__next__()
        pred = model.tangle_img(img)
        thetas[:] = model.untangle(pred, path_len, threshold, np.float32)

        glClear(GL_COLOR_BUFFER_BIT)
        glBufferData(GL_ARRAY_BUFFER, thetas.nbytes, thetas, GL_DYNAMIC_COPY)
        glDrawArrays(GL_LINES, 0, thetas.size)

        glfw.swap_buffers(window)

        frame_count += 1
        if frame_count == 10:
            show_prediction(pred)

        if delay > 0:
            time.sleep(delay/1000.0)

    image_source.close()
    glfw.terminate()
