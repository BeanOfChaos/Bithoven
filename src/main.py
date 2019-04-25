import os
from random import shuffle
from PIL import Image
import numpy as np

from Discriminator import Discriminator

IMG_SIZE = (256, 256)


if __name__ == "__main__":
    discr = Discriminator()

    cats = os.listdir('../dataset/Cat')
    dogs = os.listdir('../dataset/Dog')
    dataset = [(0, '../dataset/Dog/' + dogpic) for dogpic in dogs] \
        + [(1, '../dataset/Cat/' + catpic) for catpic in cats]

    shuffle(dataset)
    x = len(dataset) // 5
    training_set, validation_set = dataset[x:], dataset[:x]

    for type, filename in training_set:
        pic = np.array(Image.open(filename).resize(IMG_SIZE), dtype="float64")
        # normalize data
        pic /= 255
        print("TYPE: ", type)
        pred = round(discr.predict(pic))
        if pred != type:
            discr.train(type)

    # FN, FP, TN, TP
    scores = [[0, 0], [0, 0]]
    for type, filename in validation_set:
        pic = np.array(Image.open(filename).resize(IMG_SIZE), dtype="float64")
        # normalize data
        pic /= 255
        pred = round(discr.predict(pic))
        scores[pred == type][type] += 1
    print(scores)
    discr.dump_model("test.pickle")
