
import mido
import numpy as np


"""
    WARNING :
    - for now the format is a two dimension tensor s.t. a_{ij} = the value of the note on track i on time j
        '-> TODO : enhancement, make it a 3-tensor s.t. a_{ijk} = the duration of the note k on track i on time j
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
        #self.addInputLayer()
        #self.addReluLayer()
        #self.addPoolingLayer()


        l1_filter = np.zeros((1,2,3,3))
        l1_filter[0, 0, :, :] = np.array([[[-1, 0, 1], 
                                           [-1, 0, 1], 
                                           [-1, 0, 1]]])
        l1_filter[0, 1, :, :] = np.array([[[1,   1,  1], 
                                           [0,   0,  0], 
                                           [-1, -1, -1]]])

        self.addConvLayer(l1_filter)
        self.addReluLayer()
        self.addPoolingLayer()

        #pass

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
                    res[i][resi][resj] = np.max(tensor[i][np.ix_(list(range(tensi, tensi+partSize)), list(range(tensj, tensj+partSize)))])
        return res

    @staticmethod
    def relu(tensor, param):
        # relu is a simple function keeping positive values in a tensor and changing the negative ones to zero
        # it is achieved by applying the function max(e, 0) to each element of tensor
        return np.maximum(tensor, np.zeros(tensor.shape))

    @staticmethod
    def convolve(tensor, filters, featureMap):
        """
            Main convolve function. for each filter for each cell (or pixel) of the tensor, it computes the
            sum of the product of each element of the filter with each element of the corresponding sub matrix
            building a feature map
        """
        print(tensor.shape, filters.shape, featureMap.shape)
        for f in range(filters.shape[0]): # for each filter
            for i in range(featureMap.shape[1]): # line i
                for j in range(featureMap.shape[2]): # column j
                    # take a matrix from (i, j) to (i+filterSize, j+filterSize) multiply it element-wise with the filter and sum the results
                    featureMap[f][i][j] = np.sum(tensor[np.ix_(list(range(i, i+filters[f].shape[0])), list(range(j, j+filters[f].shape[1])))] * filters[f])

    @staticmethod
    def convolve2D(tensor, param):
        """
            This is the input layer, it applies a first convolution on the input and outputs the first feature map
        """
        filters = param["filters"] # an array of filters
        stride = param["stride"] # the "sliding step" of the convolution, usually 1
        featureMap = np.zeros(tuple(list(filters.shape[:-2])+list(tensor.shape[-2:]))) # init the resulting feature map
        # padding the tensor with zeros to avoid loosing data. This step may not be necessary if a small data loss is tolerated
        tensor = np.pad(tensor, ((0, filters.shape[-2] - stride), (0, filters.shape[-1] - stride)), "constant", constant_values=(0))
        # calling the convolve function
        CNN.convolve(tensor, filters, featureMap)
        return featureMap

    @staticmethod
    def convolveFeatures(tensor, param):
        """
            Standard convolution layer. It takes feature map produced by the previous layer and applies its convolution
            according to its own filters.
        """
        # see convolve2D for doc on these variables' initialization
        filters = param["filters"]
        stride = param["stride"]
        featureMap = np.zeros(tuple(list(filters.shape[:-2])+list(tensor.shape[-2:])))
        tensor = np.pad(tensor, tuple([(0,0) for _ in range(len(tensor.shape)-2)] + [(0, filters.shape[-2] - stride), (0, filters.shape[-1] - stride)]), "constant", constant_values=(0))
        # keep in mind that here, tensor is an array of feature maps produced in the previous layer
        # now, for each filter, we apply the convolution on each of the previous featuremap (one per filter of the previous layer)
        print(tensor.shape, filters.shape, featureMap.shape)
        for i in range(tensor.shape[0]):
            CNN.convolve(tensor[i], filters[i], featureMap[i])
        # once it is done, we sum the convulutions for each of the previous layer filters
        featureMap = np.sum(featureMap, axis=0)
        return featureMap


    def addInputLayer(self, filters, stride=1):
        self._layers.append(Layer(CNN.convolve2D, {"stride" : stride, "filters" : filters}))

    def addConvLayer(self, filters, stride=1):
        self._layers.append(Layer(CNN.convolveFeatures, {"stride" : stride, "filters" : filters}))

    def addReluLayer(self):
        self._layers.append(Layer(CNN.relu))

    def addPoolingLayer(self, partitionSize=2):
        self._layers.append(Layer(CNN.maxPooling, {"partitionSize" : partitionSize}))



if __name__ == '__main__':
    import skimage.data
    import matplotlib

    cnn = CNN()
    img = skimage.data.chelsea()
    res = cnn.predict(img)

    fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols=2)

    ax1[0, 0].imshow(res[0, :, :]).set_cmap("gray")
    ax1[0, 0].get_xaxis().set_ticks([])
    ax1[0, 0].get_yaxis().set_ticks([])
    ax1[0, 0].set_title("L1-Map1ReLUPool")

    ax1[0, 1].imshow(res[1, :, :]).set_cmap("gray")
    ax1[0, 1].get_xaxis().set_ticks([])
    ax1[0, 1].get_yaxis().set_ticks([])
    ax1[0, 1].set_title("L1-Map2ReLUPool")

    matplotlib.pyplot.savefig("L1.png", bbox_inches="tight")
    matplotlib.pyplot.close(fig1)

