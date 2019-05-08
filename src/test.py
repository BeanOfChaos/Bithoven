import os
import sys
from random import shuffle
import numpy as np

from Discriminator import Discriminator
from CNN.utils import loadImage, normalize



if __name__ == "__main__":


    discr = Discriminator.load_model("test100.pickle")
    cats = os.listdir('../dataset/Cat')
    dogs = os.listdir('../dataset/Dog')
    dataset = [(0, '../dataset/Dog/' + dogpic) for dogpic in dogs] \
        + [(1, '../dataset/Cat/' + catpic) for catpic in cats]

    shuffle(dataset)
    x = len(dataset) // 10
    training_set, validation_set = dataset[x:], dataset[:x]

    # FN, FP, TN, TP
    scores = [[0, 0], [0, 0]]
    for i, (type, filename) in enumerate(validation_set, 1):
        valid, pic = loadImage(filename)
        if valid:
            # normalize data
            pic = normalize(pic)
            pred = round(discr.predict(pic))
            print("\rExpected : {}; predicted : {}".format(type, pred))
            scores[pred == type][type] += 1
        print("Test {:.2%} complete".format(i/len(validation_set)), end='')
    print()
    print(scores)
    discr.dump_model("test{}.pickle".format(int(learningRate*1000)))
