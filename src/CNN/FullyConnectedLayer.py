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
		print(valuesSum)
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
		print(nodes)
		results = FullyConnectedLayer.listSoftMax(nodes)
		return results
		