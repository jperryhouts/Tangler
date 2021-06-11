import time
from typing import Iterable, Generator, Union
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import utils
from model import TangledModel

STATE = dict(SHOW_TANGLE = False,
                PAUSED = False,
                THRESHOLD = 0.0,
                RESAMPLING = 20)

class Camera():
    def __init__(self, res:int, capture_source:int=0) -> None:
        self.camera = cv2.VideoCapture(capture_source)
        self.load_crop_dimensions()
        self.res = res

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

    def img_stream(self) -> Generator[np.ndarray, None, None]:
        while self.camera.isOpened():
            success, frame = self.camera.read()

            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = img[self.y0:self.y1,self.x0:self.x1]
                img = cv2.resize(img, (self.res, self.res))
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
                    yield utils.load_img(path, res)
            self.source = gen()
        self.res = res
        self.load_cmask()

    def load_cmask(self) -> None:
        res = self.res
        X,Y = np.mgrid[-1:1:res*1j,-1:1:res*1j]
        R2 = X**2+Y**2
        self.cmask = (1*(R2<1.0)).astype(np.uint8)
        self.cmask_inv = 1-self.cmask

    def close(self) -> None:
        print("Closing ImageIterator")
        if self.type == 'webcam':
            self.cam.close()

    def __next__(self):
        img = self.source.__next__()
        img *= self.cmask
        img += self.cmask_inv*127
        return img

def show_prediction(pred, resampled):
    _, ax = plt.subplots(1,2)
    th = STATE['THRESHOLD']
    norm = colors.TwoSlopeNorm(vmin=pred.min(), vcenter=th, vmax=pred.max()) if type(th) is float else None
    im = ax[0].imshow(pred, aspect=1, cmap=plt.cm.seismic, interpolation='nearest', norm=norm)
    if type(th) is float:
        ax[0].contour(pred, levels=[th], colors='k', linewidths=0.5)
    plt.colorbar(im, ax=ax[0])
    im2 = ax[1].imshow(resampled, aspect=1, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.colorbar(im2, ax=ax[1])
    plt.show()

from subprocess import Popen, PIPE
from io import BytesIO

def img_to_ravel(img):
    sp = Popen(['/home/jmp/bin/raveler','-r','256','-f','tsv','-'], stdout=PIPE, stdin=PIPE)
    img = img.astype(np.uint8)
    so = sp.communicate(input=img.tobytes())
    pins = np.loadtxt(BytesIO(so[0])).T[0]
    return pins.astype(np.float32)

def do_demo(model_path:str, data_source, mirror:bool=False, cycle:bool=True,
        delay:int=0, path_len:int=60000, threshold:Union[str,float]='60%') -> None:

    global STATE
    STATE['THRESHOLD'] = threshold

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
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.05);
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

    pins = np.zeros(path_len, dtype=np.float32)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, pins.nbytes, pins, GL_DYNAMIC_DRAW)

    buffer_loc = glGetAttribLocation(shader, "pin")
    glEnableVertexAttribArray(buffer_loc)
    glVertexAttribPointer(buffer_loc, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    glUseProgram(shader)
    glClearColor(1., 1., 1., 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def key_callback(win, key, _A, position, _C):
        if position == 1:
            if key == glfw.KEY_Q:
                glfw.set_window_should_close(win, 1)
            elif key == glfw.KEY_S:
                STATE['SHOW_TANGLE'] = True
            elif key == glfw.KEY_SPACE:
                STATE['PAUSED'] = (not STATE['PAUSED'])
                print("Paused:",STATE['PAUSED'])
            elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                before = STATE['RESAMPLING']
                after = max(1, before + (1 if key == glfw.KEY_RIGHT else -1))
                STATE['RESAMPLING'] = after
                print(f"Resampling: {before} -> {after}")
            elif key in (glfw.KEY_DOWN, glfw.KEY_UP):
                if type(STATE['THRESHOLD']) is str:
                    before = STATE['THRESHOLD']
                    STATE['THRESHOLD'] = 0.5
                else:
                    before = f"{STATE['THRESHOLD']:0.02f}"
                    STATE['THRESHOLD'] += 0.05 if key == glfw.KEY_UP else -0.05
                print(f"Threshold: {before} -> {STATE['THRESHOLD']:0.02f}")

    glfw.set_key_callback(window, key_callback)

    frame_count = 0
    timer0 = np.zeros(100)
    timer1 = np.zeros(100)
    timer2 = np.zeros(100)
    STATE['THRESHOLD'] = 0.5
    while not glfw.window_should_close(window):
        glfw.poll_events()

        if not STATE['PAUSED']:
            frame_count += 1
            start_loop = time.perf_counter()
            img = image_source.__next__()
            begin = time.perf_counter()
            # pins[:6001] = img_to_ravel(img)
            pred = model.img_to_tangle(img)
            middle = time.perf_counter()
            resampled = utils.resample(pred, STATE['THRESHOLD'], 20)
            pins[:] = utils.untangle(resampled, path_len, 0.5, np.float32)
            end = time.perf_counter()

            glClear(GL_COLOR_BUFFER_BIT)
            glBufferData(GL_ARRAY_BUFFER, pins.nbytes, pins, GL_DYNAMIC_COPY)
            glDrawArrays(GL_LINES, 0, pins.size)

            glfw.swap_buffers(window)

            timer0[1:] = timer0[:-1]
            timer0[0] = time.perf_counter()-start_loop

            timer1[1:] = timer1[:-1]
            timer1[0] = middle-begin

            timer2[1:] = timer2[:-1]
            timer2[0] = end-middle

            if frame_count%100 == 0:
                print("Inference times (ms): [inference/untangle/loop] %d/%d/%d"%(1000*timer1.mean(), 1000*timer2.mean(), 1000*timer0.mean()))

        if STATE['SHOW_TANGLE']:
            STATE['SHOW_TANGLE'] = False
            show_prediction(pred, resampled)


        if delay > 0:
            time.sleep(delay/1000.0)


    image_source.close()
    glfw.terminate()
