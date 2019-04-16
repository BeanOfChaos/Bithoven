
import numpy as np


"""
    Input format :
       3-tensor where a_{ijk} is I don't know what (a float value) for the k'th track,
       the i'th time step and the j'th pitch
"""

class Layer:
    """
        This class wraps a layer. It contains informations usefull for the layer's computation
        as well as a reference to the function used to compute/
        Its purpose is to propose the same interface for each layer, whatever their type
    """

    def __init__(self, compFct, learnFct, parameters=None):
        self._parameters = parameters
        self._compFct = compFct
        self._learnFct = learnFct

    def compute(self, tensor):
        return self._compFct(tensor, self._parameters)

    def learn(self, loss):
        return self._learnFct(loss, self._parameters)


class CNN:
    """
        This class implements a CNN, the network is built in buildNetwork (no shit).
        TODO : implement backprop and a training function
    """

    def __init__(self, trainingSet=None):
        self._layers = []
        self.buildNetwork()
        self._trainingSet = trainingSet

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
    def convolve(tensor, param):
        """
            Convolution layer. It takes a tensor input or the feature map produced by the previous
            layer and applies its convolution according to its own filters.
        """
        filters = param["filters"] # an array of filters (3 dimensional filters)
        stride = param["stride"] # the "sliding step" of the convolution, usually 1
        featureMap = np.zeros(tuple(list(tensor.shape[:2])+[filters.shape[0]])) # init the resulting feature map
        tensor = np.pad(tensor, ((0, filters.shape[1] - stride), (0, filters.shape[2] - stride), (0,0)), "constant")
        for f in range(filters.shape[0]): # for each 3-dimensional filter
            for i in range(featureMap.shape[0]): # line i
                for j in range(featureMap.shape[2]): # column j
                    # we compute the result of the dot product between the current receptive field and the current filter (3 dimensional dot product)
                    featureMap[i][j][f] = np.tensordot(tensor[i:i+filters.shape[1], j:j+filters.shape[2], :], filters[f], axes=([0,1,2],[0,1,2]))
        return featureMap

    @staticmethod
    def maxPooling(tensor, param):
        """
            Main function of a pooling layer, its purpose is to down sample a feature map to boost
            the speed of the next layers. The method used is the maxPooling, i.e. taking the max
            value of an area as the single value retained from this area (usually areas are 2*2 squares)
        """
        partSize = param["partitionSize"] # side size of the square area
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
        return res

    @staticmethod
    def relu(tensor, param):
        # relu is a simple function keeping positive values in a tensor and changing the negative ones to zero
        # it is achieved by applying the function max(e, 0) to each element of tensor
        return np.maximum(tensor, np.zeros(tensor.shape))

    @staticmethod
    def learnConv(loss, param):
        pass

    @staticmethod
    def learnMaxPool(loss, param):
        pass

    @staticmethod
    def learnRelu(loss, param):
        pass

    @staticmethod
    def learnFullyConn(loss, param):
        pass

    def addConvLayer(self, filters, stride=1):
        self._layers.append(Layer(CNN.convolve, CNN.learnConv, {"stride" : stride, "filters" : filters}))

    def addReluLayer(self):
        self._layers.append(Layer(CNN.relu, CNN.learnRelu))

    def addPoolingLayer(self, partitionSize=2):
        self._layers.append(Layer(CNN.maxPooling, CNN.learnMaxPool, {"partitionSize" : partitionSize}))

