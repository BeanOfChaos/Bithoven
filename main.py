import os
from Pillow import Image
import numpy as np

from src import Discriminator

EPOCHS = 1000


if __name__ == "__main__":
    discr = Discriminator()

    cats = os.listdir('dataset/Cat')
    dogs = os.listdir('dataset/Dog')
    type = ''

    for i in range(EPOCHS):
        if i % 2:
            type = 1
            data = np.array(Image.open('dataset/Cat/' + cats[i//2]))
        else:
            type = 0
            data = np.array(Image.open('dataset/Dog/' + dogs[i//2 + 1]))

        pred = round(discr.predict(data))

        if pred != type:
            discr.train(type)
