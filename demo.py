import pickle as p
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from math import ceil

from keras.preprocessing.image import load_img, img_to_array


classes = ["Dog", "Cat"]
modelfile = "history.pickle"


if __name__ == "__main__":
    with open(modelfile, "rb") as f:
        model = p.load(f).model

    demo_pics = list(filter(lambda x: x.endswith(".png") or x.endswith(".jpeg") or x.endswith(".jpg"), os.listdir(".")))

    f, axarr = plt.subplots(nrows = 1, ncols = len(demo_pics))

    for i, filename in enumerate(demo_pics):
        subplt = axarr[i]

        pic = Image.open(filename)
        npic = img_to_array(load_img(filename, color_mode = 'grayscale', target_size=(256,256)))
        npic = npic.reshape([1, 256, 256, 1])
        subplt.imshow(pic)
        pred = model.predict(npic)[0][0]
        print(pred, classes[int(round(pred))])
        subplt.set_title(classes[int(round(pred))])
        subplt.axis('off')
    plt.show()
