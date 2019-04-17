import numpy as np
from CNN.AbstractLayer import Layer


class MaxPoolingLayer(Layer):

    def __init__(self, partitionSize=2, isLearning=True):
        super(MaxPoolingLayer, self).__init__(isLearning)
        self._partitionSize = partitionSize

    @staticmethod
    def maxPooling(tensor, partSize):
        """
            Main function of a pooling layer, its purpose is to down sample a feature map to boost
            the speed of the next layers. The method used is the maxPooling, i.e. taking the max
            value of an area as the single value retained from this area (usually areas are 2*2 squares)
            partSize :  side size of the square area
        """
        # computing the shape of the result tensor and building it
        resShape = tuple([int(tensor.shape[0]/partSize), int(tensor.shape[1]/partSize), tensor.shape[2]])
        res = np.zeros(resShape)
        # the tensor should be 3 dimensions, the first being the number of the filters used in the previous layer,
        # the second and third being the coordinates on the feature map
        for resi in range(resShape[-2]):
            for resj in range(resShape[-1]):
                for i in range(tensor.shape[2]):
                    # computing the index to use in tensor from the index of res
                    tensi, tensj = resi*partSize, resj*partSize
                    # computing the max value of the the sub square matrix in coordinates tensi, tensj and
                    # with a sied size equal to partSize
                    res[resi][resj][i] = np.max(tensor[tensi:tensi+partSize, tensj:tensj+partSize, i])
        return res, None # TODO return tensor used to learn data

    @staticmethod
    def learnMaxPool(loss, chosenNeurons, partSize):
        pass
    
    def compute(self, tensor):
        res, chosenNeurons = MaxPoolingLayer.maxPooling(tensor, self._partitionSize)
        self.saveData(chosenNeurons)
        return res

    def learn(self, loss):
        if self.isLearning():
            return MaxPoolingLayer.learnMaxPool(loss, self.getSavedData(), self._partitionSize)
