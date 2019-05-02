from CNN.CNN import CNN
import numpy as np
import pickle

CHANNEL_NUM = 3
LEARNING_RATE = 0.1


class Discriminator(CNN):

    def __init__(self, isLearning=True):
        super(Discriminator, self).__init__(isLearning)

    def buildNetwork(self):
        self.addConvLayer(np.random.rand(2, 7, 7, CHANNEL_NUM)*2 - 1, LEARNING_RATE)
        self.addReluLayer()
        self.addConvLayer(np.random.rand(4, 7, 7, 2)*2 - 1, LEARNING_RATE)
        self.addReluLayer()
        self.addPoolingLayer()
        self.addConvLayer(np.random.rand(2, 7, 7, 4)*2 - 1, LEARNING_RATE)
        self.addReluLayer()
        self.addPoolingLayer()
        # TODO: express the size of the layer w.r.t. convolution stride size
        # instead of hard coding it
        self.addFullyConnectedLayer(np.random.rand(6498)*2-1, LEARNING_RATE)
        # do that again

        # self.addFullyconnectedLayer(classes = 2) # is / isn't generated music

    def dump_model(self, filename):
        """Dumps the model to filename.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """Loads a previously trained (hopefully) model from filename.
           Returns it.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
