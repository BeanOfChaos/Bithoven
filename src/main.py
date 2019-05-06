import os
import sys
from random import shuffle
import numpy as np

from Discriminator import Discriminator
from CNN.utils import loadImage, normalize, LEARNING_RATE



if __name__ == "__main__":
    learningRate = LEARNING_RATE
    allowedThreads = None
    if len(sys.argv) > 1:
        learningRate = float(sys.argv[1])
    if len(sys.argv) > 2:
        allowedThreads = int(sys.argv[2])

    print("learning rate : ", learningRate)
    print("using up to {} threads".format(allowedThreads))

    discr = Discriminator(True, learningRate, allowedThreads)
    cats = os.listdir('../dataset/Cat')
    dogs = os.listdir('../dataset/Dog')
    dataset = [(0, '../dataset/Dog/' + dogpic) for dogpic in dogs] \
        + [(1, '../dataset/Cat/' + catpic) for catpic in cats]

    shuffle(dataset)
    x = len(dataset) // 5
    training_set, validation_set = dataset[x:], dataset[:x]

    for type, filename in training_set:
        print("---------------------------")
        print("image : ", filename)
        valid, pic = loadImage(filename)
        if valid:
            # normalize data
            pic = normalize(pic)
            print("TYPE: ", type)
            pred = round(discr.predict(pic))
            print("Correct!" if type == pred else "Failed!")
            discr.train(type)

    discr.unsetLearning()
    # FN, FP, TN, TP
    scores = [[0, 0], [0, 0]]
    for type, filename in validation_set:
        img = Image.open(filename)
        print("image : ", filename)
        valid, pic = loadImage(filename)
        if valid:
            # normalize data
            pic = normalize(pic)
            pred = round(discr.predict(pic))
            scores[pred == type][type] += 1
    print(scores)
    discr.dump_model("test{}.pickle".format(int(learningRate*1000)))
