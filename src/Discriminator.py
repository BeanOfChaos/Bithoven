from CNN.CNN import CNN
from CNN.utils import CHANNEL_NUM
import numpy as np
import pickle



class Discriminator(CNN):

    def __init__(self, isLearning, learningRate, allowedThreads=None):
        super(Discriminator, self).__init__(isLearning, learningRate, allowedThreads)

    def buildNetwork(self, learningRate, allowedThreads):
        self.addConvLayer(np.random.rand(16, 5, 5, CHANNEL_NUM)*2 - 1, learningRate, allowedThreads)
        self.addReluLayer()
        self.addPoolingLayer()

        self.addConvLayer(np.random.rand(8, 5, 5, 16)*2 - 1, learningRate, allowedThreads)
        self.addReluLayer()
        self.addPoolingLayer()

        self.addConvLayer(np.random.rand(4, 5, 5, 8)*2 - 1, learningRate, allowedThreads)
        self.addReluLayer()
        self.addPoolingLayer()

        self.addConvLayer(np.random.rand(2, 5, 5, 4)*2 - 1, learningRate, allowedThreads)
        self.addReluLayer()
        self.addPoolingLayer()
        # TODO: express the size of the layer w.r.t. convolution stride size
        # instead of hard coding it
        self.addFullyConnectedLayer(np.random.rand(242)*2-1, learningRate)


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
