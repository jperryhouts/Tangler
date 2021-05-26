import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import utils

def do_demo(model_path):
    model = utils.TangledModel(model_path)
    image_source = utils.ImageIterator('webcam', False, model.res)

    vertex_src = '''
    in float pin;
    void main() {
        float PI = 3.1415926535897932384626433832795;
        float n_pins = ''' + str(model.n_pins) + ''';
        float theta = 2.0 * PI * pin.x / n_pins;
        float x = sin(theta), y = cos(theta);
        gl_Position = vec4(x, y, 0.0, 1.0);
    }
    '''

    fragment_src = '''
    void main() {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.1);
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

    path = np.zeros(model.max_path_len, dtype=np.float32)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, path.nbytes, path, GL_DYNAMIC_DRAW)

    buffer_loc = glGetAttribLocation(shader, "pin")
    glEnableVertexAttribArray(buffer_loc)
    glVertexAttribPointer(buffer_loc, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    glUseProgram(shader)
    glClearColor(1., 1., 1., 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        img = image_source.__next__()
        path[:] = model.predict(img).astype(np.float32)

        glClear(GL_COLOR_BUFFER_BIT)
        glBufferData(GL_ARRAY_BUFFER, path.nbytes, path, GL_DYNAMIC_COPY)
        glDrawArrays(GL_LINES, 0, path.size)

        glfw.swap_buffers(window)

    image_source.close()
    glfw.terminate()