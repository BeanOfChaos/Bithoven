from math import exp, pow
import numpy as np
from AbstractLayer import Layer

class FullyConnectedLayer(Layer):
	
	def __init__(self, filters, stride=1, isLearning=True):
		super(FullyConnectedLayer, self).__init__(isLearning)
		self._filters = filters
		self._stride = stride
	
	def sigmoid(value):
		"""
		Basic Sigmoid calculation
		"""
		return exp(value)/(exp(value) + 1)
	
		
	@staticmethod
	def connect(vector, filter):
		"""
		Does the dot product between the input vector and the filter.
		Vector is a   1 x n array
		Filter is a   n x 1 array
		result is a   1 x 1 array for the two nodes of the fully connected layer, on which SoftMax is applied
		"""
		node = np.dot(vector, filter)
		result = FullyConnectedLayer.sigmoid(node)
		return result
		

	def calculateLeastSquares(prediction, actual):
		return 1/2 * pow(prediction-actual, 2)
		
		
	@staticmethod
	def learnConv(previousLayer, receivedInput, filter, learningRate, prediction, actual):
		"""
		Function computing the loss of the previous layer and the updated filter.
		There is only one filter
		Previous Layer : the simple values of the previous layer's nodes
		Received input : sum (xi * wi)
		"""
		
		loss = calculateLeastSquaresVector(prediction, actual)
		
		previousLayerLoss = np.zeros(receivedInput.shape) # contains the loss of the previous layer
		filtersCorrection = np.zeros(filter.shape) # will be used to compute the updated filters

		for i in range(filter.shape[0]):  #for i along the height
			filtersCorrection[i] = loss * (previousLayer[i] * exp(receivedInput))/pow((exp(receivedInput)+1), 2) 
			filter = filter - learningRate*filtersCorrection
			
			previousLayerLoss[i] = loss * (filter[i] * exp(receivedInput))/pow((exp(receivedInput)+1),2)
		
		
		return previousLayerLoss, filter


