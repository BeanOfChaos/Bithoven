import numpy as np
from PIL import Image


LEARNING_RATE = 0.01
CHANNEL_NUM = 3
IMG_SIZE = (256, 256)


def loadImage(filename):
    valid, img = True, None
    try:
        img = Image.open(filename)
        img.verify()
    except Exception as e:
        valid = False
    else:
        img = np.array(Image.open(filename).resize(IMG_SIZE), dtype="float64")
        if img.shape != (IMG_SIZE[0], IMG_SIZE[1], CHANNEL_NUM):
            valid = False
    finally:
        return valid, img


def generateWeights(inSize, outSize):
    return np.random.rand(inSize, outSize) * np.sqrt(1/inSize)


def generateFilters(nfilter, nline, ncol, nchan=CHANNEL_NUM):
    # size = nline * ncol * nchan
    return (np.random.rand(nfilter, nline, ncol, nchan) *2 -1) #  * np.sqrt(1/size)


def normalize(pic):
    pic -= np.mean(pic)
    pic /= np.std(pic)
    return pic


def sigmoid(value, derivative=False):
    """
    Basic Sigmoid calculation
    """
    # return np.exp(value)/(np.exp(value) + 1)
    if derivative:
        sigm = sigmoid(value)
        return sigm * (1 - sigm)
    else:
        return 1/(np.exp(-value) + 1)


def reLu(x, derivative=False):
    """Rectified linear unit (ReLu) function
    """
    if derivative:
        return x > 0  # True == 1 and False == 0
    else:
        return x * (x > 0)
