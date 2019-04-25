from CNN.CNN import CNN
import numpy as np
import pickle

CHANNEL_NUM = 3


class Discriminator(CNN):

    def __init__(self, isLearning=True):
        super(Discriminator, self).__init__(isLearning)

    def buildNetwork(self):
        self.addConvLayer(np.random.rand(2, 7, 7, CHANNEL_NUM), 0.01)
        self.addReluLayer()
        #self.addConvLayer(np.random.rand(4, 7, 7, 8), 0.01)
        #self.addReluLayer()
        self.addPoolingLayer(partitionSize=4)
        #self.addConvLayer(np.random.rand(2, 7, 7, 4), 0.01)
        #self.addReluLayer()
        #self.addPoolingLayer()
        # TODO: express the size of the layer w.r.t. convolution stride size
        self.addFullyConnectedLayer(np.random.rand(7688), 0.01)
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
