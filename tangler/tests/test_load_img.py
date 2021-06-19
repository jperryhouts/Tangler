import os
import numpy as np
import pickle
from tangler.img_utils import load_img

DATA_DIR = os.path.join(os.path.dirname(os.path.relpath(__file__)), 'test_files')

def test_load_img():
    datafiles = (
        os.path.join(DATA_DIR, 'n03417042_553.jpg'),
        os.path.join(DATA_DIR, 'n03417042_5406.jpg'),
        os.path.join(DATA_DIR, 'n03417042_5424.jpg'),
        os.path.join(DATA_DIR, 'n03417042_5449.jpg'))
    for fname in datafiles:
        img = load_img(fname, 256)
        assert type(img) is np.ndarray
        assert img.shape == (256,256)

        compare = fname.rsplit('.',1)[0]+'.p'
        # with open(compare, 'wb') as pimg:
        #     pickle.dump(img, pimg)

        with open(compare, 'rb') as pimg:
            cimg = pickle.load(pimg)
            assert (img == cimg).all()
