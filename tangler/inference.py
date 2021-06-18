from typing import Iterable, Optional
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow as tf
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from . import utils
from .image_handling import init_image_source
# from .model import TangledModel
from .simple_model import SimpleModel, encoder_stack

class AppState():
    def __init__(self, threshold:float=0.0, resampling:int=20, paused:bool=False):
        self.show_tangle = False
        self.paused = paused
        self.threshold = threshold
        self.resampling = resampling
        self.raw_feed = False

    def handle_input(self, win, key, _A, position, _C):
        if position == 1:
            if key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, 1)
            elif key == glfw.KEY_S:
                self.show_tangle = True
            elif key == glfw.KEY_SPACE:
                self.raw_feed = (not self.raw_feed)
                print("Raw webcam feed:",self.raw_feed)
            elif key == glfw.KEY_P:
                self.paused = (not self.paused)
                print("Paused:",self.paused)
            elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                before = self.resampling
                after = max(1, before + (1 if key == glfw.KEY_RIGHT else -1))
                self.resampling = after
                print(f"Resampling: {before} -> {after}")
            elif key in (glfw.KEY_DOWN, glfw.KEY_UP):
                if type(self.threshold) is str:
                    before = self.threshold
                    self.threshold = 0.5
                else:
                    before = f"{self.threshold:0.02f}"
                    self.threshold += 0.05 if key == glfw.KEY_UP else -0.05
                print(f"Threshold: {before} -> {self.threshold:0.02f}")

    def show_prediction(self, pred, resampled):
        self.show_tangle = False

        _, ax = plt.subplots(1,2)
        th = self.threshold
        norm = colors.TwoSlopeNorm(vmin=pred.min(), vcenter=th, vmax=pred.max()) if type(th) is float else None
        im = ax[0].imshow(pred, aspect=1, cmap=plt.cm.seismic, interpolation='nearest', norm=norm)
        if type(th) is float:
            ax[0].contour(pred, levels=[th], colors='k', linewidths=0.5)
        plt.colorbar(im, ax=ax[0])
        im2 = ax[1].imshow(resampled, aspect=1, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.colorbar(im2, ax=ax[1])
        plt.show()

def demo(model_path:str, data_source:str, inputs:Optional[Iterable]=None,
        mirror:bool=False, cycle:bool=True, delay:int=0, path_len:int=35000,
        threshold:float=-2.5, res:int=512, webcam:Optional[str]=None,
        aspect:float=1.0, show_stats:bool=False) -> None:

    res_x, res_y = int(aspect*res), res

    app_state = AppState(threshold)

    # model = tf.keras.models.load_model(model_path)

    n_pins = 256

    encoder = encoder_stack()
    model = SimpleModel(encoder)

    try:
        encoder.trainable = False
        model.load_weights(model_path)
    except ValueError:
        encoder.trainable = True
        model.load_weights(model_path)

    # model = TangledModel()
    # model.load_weights(model_path)

    image_source = init_image_source(data_source, inputs, mirror, cycle)

    if webcam is not None:
        import pyfakewebcam
        webcam = pyfakewebcam.FakeWebcam(webcam, res_x, res_y)


    vertex_src = f'''
    in float pin;
    void main() {{
        float PI = 3.1415926535897932384626433832795;
        float n_pins = {n_pins};
        float theta = 2.0 * PI * pin.x / n_pins;
        float x = sin(theta), y = cos(theta);
        gl_Position = vec4(x/{aspect}, -y, 0.0, 1.0);
    }}
    '''

    thread_wt = 75e-6
    fragment_src = f'''
    void main() {{
        gl_FragColor = vec4(0.0, 0.0, 0.0, {res*thread_wt});
    }}
    '''

    if not glfw.init():
        raise Exception("Cannot initialize glfw")

    window = glfw.create_window(res_x, res_y, "Tangler Demo", None, None)

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

    glfw.set_key_callback(window, app_state.handle_input)

    frame_count = 0
    timer0 = np.zeros(100)
    timer1 = np.zeros(100)
    timer2 = np.zeros(100)
    while not glfw.window_should_close(window):
        glfw.poll_events()

        frame_count += 1
        src_cropped_rgb = image_source.read()
        if src_cropped_rgb is None:
            break

        timer0[1:] = timer0[:-1]
        timer0[0] = time.perf_counter()

        glClear(GL_COLOR_BUFFER_BIT)

        if not app_state.paused:
            _model_input_res = 256
            preprocessed = image_source.resize(src_cropped_rgb, _model_input_res)
            preprocessed = image_source.rgb2gray(preprocessed)
            predicted = utils.img_to_tangle(model, preprocessed)

            timer1[1:] = timer1[:-1]
            timer1[0] = time.perf_counter()

            tangle = utils.resample(predicted, app_state.threshold, app_state.resampling)
            pins[:] = utils.untangle(tangle, path_len, 0.5, np.float32)
            # pins[:6001] = utils.img_to_ravel(preprocessed)

            timer2[1:] = timer2[:-1]
            timer2[0] = time.perf_counter()

            glBufferData(GL_ARRAY_BUFFER, pins.nbytes, pins, GL_DYNAMIC_COPY)

        glDrawArrays(GL_LINES, 0, pins.size)
        glfw.swap_buffers(window)

        if frame_count%100 == 0 and show_stats:
            inf_time = int(1000*(timer1-timer0).mean())
            u_time = int(1000*(timer2-timer1).mean())
            frame_rate = (timer0.size-1)/(timer0[0]-timer0[-1])
            print(f"{frame_rate:0.03f} fps: {inf_time} ms / {u_time} ms [inference/untangle]")

        if webcam is not None:
            if app_state.raw_feed:
                frame = np.zeros((res_y, res_x, 3), dtype=np.uint8)
                c1 = (res_x-res)//2
                frame[:,c1:c1+res,:] = image_source.resize(src_cropped_rgb, res)
            else:
                pix_buffer = glReadPixels(0, 0, res_x, res_y, GL_RGB, GL_UNSIGNED_BYTE)
                frame = np.frombuffer(pix_buffer, dtype=np.uint8, count=res_x*res_y*3)
                frame = frame.reshape((res_y,res_x,3))
                frame = frame[::-1,:]

            webcam.schedule_frame(frame)

        if app_state.show_tangle:
            app_state.show_prediction(predicted, tangle)


        if delay > 0:
            time.sleep(delay/1000.0)


    image_source.close()
    glfw.terminate()
