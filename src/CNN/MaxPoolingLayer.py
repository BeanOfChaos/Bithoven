import numpy as np
from CNN.AbstractLayer import Layer


class MaxPoolingLayer(Layer):

    def __init__(self, partitionSize=2, isLearning=True):
        super(MaxPoolingLayer, self).__init__(isLearning)
        self._partitionSize = partitionSize

    @staticmethod
    def maxPooling(tensor, partSize):
        """
            Main function of a pooling layer, its purpose is to down sample
            a feature map to boost the speed of the next layers.
            The method used is the maxPooling, i.e. taking the max value
            of an area as the single value retained from this area
            (usually areas are 2*2 squares).
            partSize :  side size of the square area
        """
        # computing the shape of the result tensor and building it
        resShape = tuple([int(tensor.shape[0]/partSize),
                          int(tensor.shape[1]/partSize),
                          tensor.shape[2]])
        res = np.zeros(resShape)
        # the tensor should be 3 dimensions,
        # the first being the number of the filters used in the previous layer,
        # the second and third being the coordinates on the feature map
        for resi in range(resShape[0]):
            for resj in range(resShape[1]):
                for k in range(resShape[2]):
                    # computing the index to use in tensor from the index of res
                    tensi, tensj = resi*partSize, resj*partSize
                    # computing the max value of the the sub square matrix
                    # in coordinates (tensi, tensj), with side size partSize
                    res[resi, resj, k] = np.max(tensor[tensi:tensi+partSize,
                                                       tensj:tensj+partSize,
                                                       k])
        return res  # TODO return tensor used to learn data

    @staticmethod
    def learnMaxPool(loss, savedData, partSize):
        """
            Learning function of max poolingself.
            Sends back the loss to the neurons chosen in the forward pass
        """
        # savedData = (receivedInput, sentData)
        previousLayerLoss, sentData = savedData

        for lossi in range(loss.shape[0]):
            for lossj in range(loss.shape[1]):
                for k in range(loss.shape[2]):
                    inputi, inputj = lossi * partSize, lossj * partSize
                    # max values sent are used to find their position in the input and replace
                    # them by the loss, other values are replaced by 0
                    previousLayerLoss[inputi:inputi + partSize,
                                      inputj:inputj + partSize,
                                      k] \
                        = np.where(previousLayerLoss[inputi:inputi+partSize,
                                                     inputj:inputj + partSize,
                                                     k]
                                   == sentData[lossi, lossj, k],
                                   loss[lossi, lossj, k],
                                   0)
        return previousLayerLoss

    def compute(self, tensor):
        """
            Wraps the computation static method
        """
        res = MaxPoolingLayer.maxPooling(tensor, self._partitionSize)
        self.saveData((tensor, res))
        return res

    def learn(self, loss):
        """
            Wraps the learning static method and update the filters
        """
        if self.isLearning():
            return MaxPoolingLayer.learnMaxPool(loss,
                                                self.getSavedData(),
                                                self._partitionSize)
