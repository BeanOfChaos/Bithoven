from CNN.AbstractLayer import Layer
import numpy as np

from multiprocessing import Pool
from os import cpu_count
from math import ceil


class ConvolutionLayer(Layer):

    def __init__(self, filters, learningRate, stride=1, isLearning=True, allowedThreads=None):
        super(ConvolutionLayer, self).__init__(isLearning)
        self._learningRate = learningRate
        self._filters = filters
        self._stride = stride
        self._allowedThreads = allowedThreads or cpu_count()


    def parallelize(self, func, param):
        pool = Pool(self._allowedThreads)
        res = pool.starmap_async(func, param, chunksize=ceil(len(param)/self._allowedThreads))
        pool.close()
        pool.join()
        return res.get()

    @staticmethod
    def convolveFilter(tensor, filters, fIndex, stride):
        shape = list(tensor.shape[:-1])
        shape = [(shape[i] - filters.shape[i+1]) // (stride)
                for i in range(len(shape))] + [1]
        featureMap = np.zeros(tuple(shape))
        for i in range(0, featureMap.shape[0], stride):  # line i
            for j in range(0, featureMap.shape[1], stride):  # column j
                # we compute the result of the dot product between:
                # (1) the current receptive field, and
                # (2) the current filter (3 dimensional dot product)
                featureMap[i][j][0] \
                    = np.tensordot(tensor[i:i+filters.shape[1], j:j+filters.shape[2], :], filters[fIndex], axes=((0, 1, 2), (0, 1, 2))) / filters[0].size
        return (fIndex, featureMap)


    def convolve(self, tensor):
        """
            Convolution layer.
            It takes a tensor input or the feature map produced by the previous
            layer and applies its convolution according to its own filters.
            stride : the "sliding step" of the convolution, usually 1
            filters : an array of filters (3 dimensional filters)
        """
        # init the resulting feature map
        shape = list(tensor.shape[:-1])
        shape = [(shape[i] - self._filters.shape[i+1]) // (self._stride)
                for i in range(len(shape))] + [self._filters.shape[0]]
        featureMap = np.zeros(tuple(shape))
        res = self.parallelize(ConvolutionLayer.convolveFilter, [(tensor, self._filters, i, self._stride) for i in range(self._filters.shape[0])])
        for f, partFMap in res:
            featureMap[:][:][f] = partFMap[:][:][0]
        return featureMap

    @staticmethod
    def learnConv(loss, receivedInput, filters, stride, learningRate):
        """
            Function computing the loss of the previous layer and the updated filters.
            The received loss is computed in the next layer and sent here through backprop.
        """
        # contains the loss of the previous layer
        previousLayerLoss = np.zeros(receivedInput.shape)
        # will be used to compute the updated filters
        filtersCorrection = np.zeros(filters.shape)
        for i in range(filters.shape[0]):  # for each filter
            for j in range(0, filters.shape[1], stride):  # for i along the height
                for k in range(0, filters.shape[2], stride):  # for j along the width
                    # computing dL/dinput and dL/dW
                    previousLayerLoss[j:j+filters.shape[1],
                                      k:k+filters.shape[2], :] \
                                        += loss[j, k, i] * filters[i]
                    filtersCorrection[i] += loss[j, k, i] \
                        * receivedInput[j:j+filters.shape[1],
                                        k:k+filters.shape[2],
                                        :]
        # returns the previous layer's loss and the updated filters
        return previousLayerLoss, filters - learningRate * filtersCorrection

    def compute(self, tensor):
        """
            Wraps the computation static method
        """
        self.saveData(tensor)
        return self.convolve(tensor)

    def learn(self, loss):
        """
            Wraps the learning static method and update the filters
        """
        if self.isLearning():
            res, self._filters = ConvolutionLayer.learnConv(loss,
                                                            self.getSavedData(),
                                                            self._filters,
                                                            self._stride,
                                                            self._learningRate)
            return res
