from math import exp
import numpy as np
from AbstractLayer import Layer

class FullyConnectedLayer(Layer):
	
	def __init__(self, filters, stride=1, isLearning=True):
		super(FullyConnectedLayer, self).__init__(isLearning)
		self._filters = filters
		self._stride = stride
	
	def softMax(value, valuesSum):
		"""
		Basic SoftMax calculation
		"""
		return exp(value)/valuesSum
	
	def listSoftMax(lst):
		"""
		Calculate SoftMax for a list of values
		"""
		valuesSum = 0
		for val in lst : 
			valuesSum += exp(val)
		results = []
		for val in lst:
			results.append(FullyConnectedLayer.softMax(val, valuesSum))
		return results
		
	@staticmethod
	def connect(vector, filter):
		"""
		Does the dot product between the input vector and the filter.
		Vector is a   1 x n array
		Filter is a   n x 2 array
		result is a   1 x 2 array for the two nodes of the fully connected layer, on which SoftMax is applied
		"""
		nodes = np.dot(vector, filter)
		results = FullyConnectedLayer.listSoftMax(nodes)
		return results
		

	def calculateCrossEntropy(prediction, actual):
		return actual * math.log(prediction)
	
	def calculateCrossEntropyVector(predicted, actuals):
		res = [FullyConnectedLayer.calculateCrossEntropy(predicted[i], actuals[i]) for i in range(len(predicted))]
		return -(np.sum(res))
		
	
		
	@staticmethod
	def learnConv(loss, previousLayer, receivedInput, filter, learningRate):
		"""
		Function computing the loss of the previous layer and the updated filter.
		There is only one filter
		"""
		previousLayerLoss = np.zeros(receivedInput.shape) # contains the loss of the previous layer
        filtersCorrection = np.zeros(filter.shape) # will be used to compute the updated filters
		
		for i in range(filter.shape[0]):  #for i along the height
			for j in range(filter.shape[1]):  #for j along the width
				filtersCorrection[i][j] = loss[j] * 
				(previousLayer[i]*math.exp(receivedInput[0])*math.exp(receivedInput[1]))/(math.exp(receivedInput[0])+math.exp(receivedInput[1]))**2 
				#derivation of softmax formula with cross entropy. dE/dW
				
				#TODO !!!! Previous layer loss
		
		filter = filter - learningRate*filtersCorrection
		return filter