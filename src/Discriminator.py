from CNN.CNN import CNN
from utils import dataset, toTensor
import numpy as np

CHANNEL_NUM = 5

#TODO manage saved filters with pickle


class Discriminator(CNN):

    def __init__(self, isTraining=False, trainingSet=None):
        super(Discriminator, self).__init__(isTraining, trainingSet)

    def buildNetwork(self):
        self.addConvLayer(np.random.rand(1, 7, 7, CHANNEL_NUM), 0.1)
        self.addReluLayer()
        self.addPoolingLayer()
        #do that again

        #self.addFullyconnectedLayer(classes = 2) # is generated music / or not



if __name__ == '__main__':
    test = Discriminator()
    for song in dataset("./lpd_cleansed/"):
        tensor = toTensor(song)
        if tensor.shape[2] == 5:
            print("input tensor :", tensor.shape)
            print(test.predict(tensor).shape)

