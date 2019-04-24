import numpy as np
from CNN.AbstractLayer import Layer


class ReluLayer(Layer):

    def __init__(self, isLearning=True):
        super(ReluLayer, self).__init__(isLearning)

    def compute(self, tensor):
        # relu is a simple function keeping positive values in a tensor,
        # and changing the negative ones to zero.
        self.saveData(tensor)
        return np.maximum(tensor, np.zeros(tensor.shape))

    def learn(self, loss):
        """
            the derivative of Relu is 1 where x>0, 0 elsewhere
            the loss of the previous layer is the derivative of Relu,
            multiplied by the loss of this layer.
        """
        return loss * np.where(self.getSavedData() > 0, 1, 0)
