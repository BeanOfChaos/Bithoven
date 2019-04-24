import os
from random import shuffle
from PIL import Image
import numpy as np

from Discriminator import Discriminator


if __name__ == "__main__":
    discr = Discriminator()

    cats = os.listdir('../dataset/Cat')
    dogs = os.listdir('../dataset/Dog')
    dataset = [(1, '../dataset/Cat/' + catpic) for catpic in cats] \
        + [(0, '../dataset/Dog/' + dogpic) for dogpic in dogs]

    shuffle(dataset)
    x = len(dataset) // 5
    training_set, validation_set = dataset[x:], dataset[:x]

    for type, filename in training_set:
        pic = np.array(Image.open(filename))
        pred = round(discr.predict())
        if pred != type:
            discr.train(pred)

    # FN, FP, TN, TP
    scores = [[0, 0], [0, 0]]
    for type, filename in validation_set:
        pic = np.array(Image.open(filename))
        pred = round(discr.predict())
        scores[pred == type][type] += 1
