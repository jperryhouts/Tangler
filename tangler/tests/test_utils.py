import numpy as np
from tangler import utils

def test_target_to_complex():
    a = np.random.random((256,256,2))
    b = utils.target_to_complex(a, 256)
    for i,j in np.random.randint(0,256,(5,2)):
        real, imag = a[i,j,:]
        cmplex = b[i,j]
        assert cmplex == real + imag*1j

def test_target_to_tangle():
    size = 256*256
    for i in range(4):
        a = size // 2**i
        b = size // a
        assert a*b == size

        target = np.random.random((a,b))
        tangle = utils.target_to_tangle(target)

        assert (tangle.flatten() == target.flatten()).all()


def test_predict():
    class DummyModel():
        def __init__(self, res):
            self.res = res

        def predict(self, img):
            self.latest = img
            return np.zeros((self.res,self.res))

    model = DummyModel(256)
    img = np.random.randint(0,256,(256,256),dtype=np.uint8)
    tangle = utils.img_to_tangle(model, img)
    assert (tangle == 0).all()
    assert model.latest.shape == (1,256,256,1)
    assert model.latest.dtype is np.dtype('float32')
    assert model.latest.min() >= -1 and model.latest.min() <= 1
    assert model.latest.max() >= -1 and model.latest.max() <= 1
    received = model.latest.flatten()
    expected = (img.flatten()-127.5)/127.5
    assert np.isclose(received, expected).all()


def test_resampling():
    def check(iters, expected):
        original = np.ones((256,256))
        np.random.seed(42)
        resampled = utils.resample(original, 0.5, iters)
        n_dropped = (1*(resampled == 0)).sum()
        assert n_dropped == expected

    test_combos = [(0, 65536), (1, 62294), (25, 18122)]
    for iters, exp in test_combos:
        check(iters, exp)
