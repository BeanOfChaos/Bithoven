
import mido
import numpy as np


"""
    Input format :
       3-tensor where a_{ijk} is I don't know what (a float value) for the i'th track,
       the j'th time step and the k'th pitch
"""

class Layer:
    """
        This class wraps a layer. It contains informations usefull for the layer's computation
        as well as a reference to the function used to compute/
        Its purpose is to propose the same interface for each layer, whatever their type
    """

    def __init__(self, compFct,  parameters=None):
        self._parameters = parameters
        self._compFct = compFct

    def compute(self, tensor):
        return self._compFct(tensor, self._parameters)



class CNN:
    """
        This class implements a CNN, the network is built in buildNetwork (no shit).
        TODO : implement backprop and a training function
    """

    def __init__(self, trainingSet=None):
        self._layers = []
        self.buildNetwork()
        self._trainingSet = trainingSet
        #self.train()
        #TODO manage saved filters with pickle

    def buildNetwork(self):
        """
            Here is built the network, the layers are stacked, the first one being the input layer,
            the last one being the output layer. The ouput of a layer is the input of the next
        """
        pass

    def predict(self, inputTensor):
        """
            Main "predicting" method of the CNN, based on what is learned and the architecture of
            the network, it outputs a result
        """
        currentTensor = inputTensor
        for layer in self._layers:
            currentTensor = layer.compute(currentTensor)
        return currentTensor

    @staticmethod
    def maxPooling(tensor, param):
        """
            Main function of a pooling layer, its purpose is to down sample a feature map to boost
            the speed of the next layers. The method used is the maxPooling, i.e. taking the max
            value of an area as the single value retained from this area (usually areas are 2*2 squares)
        """
        partSize = param["partitionSize"] # side size of the square area
        # computing the shape of the result tensor and building it
        resShape = tuple(list(tensor.shape[:-2]) + [int(tensor.shape[-2]/partSize), int(tensor.shape[-1]/partSize)])
        res = np.zeros(resShape)
        # the tensor should be 3 dimensions, the first being the number of the filters used in the previous layer,
        # the second and third being the coordinates on the feature map
        for i in range(tensor.shape[0]):
            for resi in range(resShape[-2]):
                for resj in range(resShape[-1]):
                    # computing the index to use in tensor from the index of res
                    tensi, tensj = resi*partSize, resj*partSize
                    # computing the max value of the the sub square matrix in coordinates tensi, tensj and
                    # with a sied size equal to partSize
                    res[i][resi][resj] = np.max(tensor[i, tensi:tensi+partSize, tensj:tensj+partSize])
        return res

    @staticmethod
    def relu(tensor, param):
        # relu is a simple function keeping positive values in a tensor and changing the negative ones to zero
        # it is achieved by applying the function max(e, 0) to each element of tensor
        return np.maximum(tensor, np.zeros(tensor.shape))

    @staticmethod
    def convolveFeatures(tensor, param):
        """
            Convolution layer. It takes a tensor input or the feature map produced by the previous
            layer and applies its convolution according to its own filters.
        """
        filters = param["filters"] # an array of filters (3 dimensional filters)
        stride = param["stride"] # the "sliding step" of the convolution, usually 1
        featureMap = np.zeros(tuple([filters.shape[0]]+list(tensor.shape[-2:]))) # init the resulting feature map
        tensor = np.pad(tensor, ((0,0), (0, filters.shape[2] - stride), (0, filters.shape[3] - stride)), "constant")
        for f in range(filters.shape[0]): # for each 3-dimensional filter
            for i in range(featureMap.shape[1]): # line i
                for j in range(featureMap.shape[2]): # column j
                    # we compute the result of the dot product between the current receptive field and the current filter (3 dimensional dot product)
                    featureMap[f][i][j] = np.tensordot(tensor[:, i:i+filters.shape[-2], j:j+filters.shape[-1]], filters[f], axes=([0,1,2],[0,1,2]))
        return featureMap


    def addInputLayer(self, filters, stride=1):
        self._layers.append(Layer(CNN.convolve2D, {"stride" : stride, "filters" : filters}))

    def addConvLayer(self, filters, stride=1):
        self._layers.append(Layer(CNN.convolveFeatures, {"stride" : stride, "filters" : filters}))

    def addReluLayer(self):
        self._layers.append(Layer(CNN.relu))

    def addPoolingLayer(self, partitionSize=2):
        self._layers.append(Layer(CNN.maxPooling, {"partitionSize" : partitionSize}))

