import glob, itertools, os
from typing import Union, Iterable, Generator
from PIL import Image, ImageOps
import numpy as np
import cv2

def load_img(src: str, res: int, grayscale:bool=True) -> np.array:
    img = Image.open(src)
    if grayscale:
        img = ImageOps.grayscale(img)

    crop = None
    if img.size[0] != img.size[1]:
        cx, cy = (img.size[0]//2, img.size[1]//2)
        size2 = min(img.size)//2
        crop = (cx-size2, cy-size2, cx+size2, cy+size2)

    if res != -1:
        if (img.size[0] != res) or (crop is not None):
            img = img.resize((res, res), box=crop)

    return np.array(img)

class ImageStream():
    def __init__(self, source:Union[int,str,Iterable], loop:bool) -> None:
        pass

    def generator(self) -> Generator[np.ndarray, None, None]:
        yield None

    def close(self) -> None:
        pass

class Capture(ImageStream):
    def __init__(self, source:Union[int,str], loop=False) -> None:
        self.loop = loop
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.load_crop_dimensions()

    def close(self) -> None:
        print('Releasing camera')
        self.cap.release()

    def load_crop_dimensions(self) -> None:
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == h:
            self.y0, self.x0 = [0,0]
            self.y1, self.x1 = [h-1,w-1]
        else:
            imres = min(w,h)
            wc, hc = w//2, h//2
            self.y0, self.x0 = (hc-imres//2, wc-imres//2)
            self.y1, self.x1 = (hc+imres//2, wc+imres//2)

    def generator(self) -> Generator[np.ndarray, None, None]:
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = img[self.y0:self.y1,self.x0:self.x1]
                yield img
            elif self.loop:
                try:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                except:
                    print("Error while reloading capture source")
                    yield None
            else:
                yield None

class FileStream(ImageStream):
    def __init__(self, paths:Iterable, loop:bool=True):
        self.paths = itertools.cycle(paths) if loop else itertools.chain(paths)

    def generator(self) -> Generator[np.ndarray, None, None]:
        for path in self.paths:
                yield load_img(path, res=-1, grayscale=False)

    def close(self):
        pass

class ImageSource():
    def __init__(self, source:ImageStream, mirror:bool=False) -> None:
        self.mirror = mirror
        self.source = source
        self.generator = source.generator()

    def close(self) -> None:
        print("Closing ImageIterator")
        self.source.close()

    def read(self):
        ## Returnes square-cropped RGB image from current source.
        img = self.generator.__next__()
        if img is None:
            return None

        if self.mirror:
            img = img[:,::-1]
        self.latest = img
        return img

    @staticmethod
    def resize(img, res):
        return cv2.resize(img, (res,res))

    @staticmethod
    def rgb2gray(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def gray2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def stretch_contrast(img):
        img = img.astype(np.float64)
        img = (img - img.mean())/(3*img.std()+0.01) + 0.5
        np.clip(img, 0, 1, img)
        img = (img-img.min())/img.ptp()
        return (255*img).astype(np.uint8)

def init_image_source(source:str, inputs:Iterable=[0], mirror=False, cycle=False) -> ImageSource:
    if source == "files":
        sources = []
        for src in inputs:
            if os.path.isdir(src):
                sources += glob.glob(os.path.join(src, '*.jpg'))
                sources += glob.glob(os.path.join(src, '*.JPEG'))
            else:
                sources.append(src)
        for src in sources:
            assert os.path.isfile(src), \
                f'Not a valid input source: {src}'
        stream = FileStream(sources, cycle)
    elif source == 'webcam':
        stream = Capture(int(inputs[0]), False)
    elif source == 'video':
        stream = Capture(inputs[0], cycle)

    return ImageSource(stream, mirror)