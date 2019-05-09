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
        print("learning rate : ", learningRate)
    if len(sys.argv) > 2:
        allowedThreads = int(sys.argv[2])
        print("using up to {} threads".format(allowedThreads))


    discr = Discriminator(True, learningRate, allowedThreads)
    cats = os.listdir('../dataset/Cat')
    dogs = os.listdir('../dataset/Dog')
    dataset = [(0, '../dataset/Dog/' + dogpic) for dogpic in dogs] \
        + [(1, '../dataset/Cat/' + catpic) for catpic in cats]

    shuffle(dataset)
    x = len(dataset) // 10
    training_set, validation_set = dataset[x:], dataset[:x]

    for i, (type, filename) in enumerate(training_set, 1):
        valid, pic = loadImage(filename)
        if valid:
            # normalize data
            pic = normalize(pic)
            pred = discr.predict(pic)
            pred = np.round(pred)
            error = discr.train(type)
            print("\rImage {}/{} : {} (exp. {}; pred. {}) Error : {}".format(i, len(training_set), "Correct" if type == pred else "Failed", type, pred, error))
            print("Training {:.2%} complete.".format(i/len(training_set)), end='')
    print()

    discr.unsetLearning()

    # FN, FP, TN, TP
    scores = [[0, 0], [0, 0]]
    for i, (type, filename) in enumerate(validation_set, 1):
        valid, pic = loadImage(filename)
        if valid:
            # normalize data
            pic = normalize(pic)
            pred = round(discr.predict(pic))
            scores[pred == type][type] += 1
        print("\rValidation {:.2%} complete".format(i/len(validation_set)), end='')
    print()
    print(scores)
    discr.dump_model("test{}.pickle".format(int(learningRate*1000)))
