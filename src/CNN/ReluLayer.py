import numpy as np
from CNN.AbstractLayer import Layer


class ReluLayer(Layer):

    def __init__(self, isLearning=True):
        super(ReluLayer, self).__init__(isLearning)


    @staticmethod
    def learnRelu(loss, receivedInput):
        """
            the derivative of Relu is 1 where x>0, 0 elsewhere
            the loss of the previous layer is the derivative of Relu multiplied by the loss of this layer
        """
        previousLayerLoss = np.where(receivedInput>0, 1, 0)
        return loss * previousLayerLoss

    def compute(self, tensor):
        # relu is a simple function keeping positive values in a tensor and changing the negative ones to zero
        # it is achieved by applying the function max(e, 0) to each element of tensor
        self.saveData(tensor)
        return np.maximum(tensor, np.zeros(tensor.shape))

    def learn(self, loss):
        return ReluLayer.learnRelu(loss, self.getSavedData())
