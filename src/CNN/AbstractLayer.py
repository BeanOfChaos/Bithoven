class Layer:

    def __init__(self, isLearning):
        self._isLearning = isLearning
        self._savedData = None

    def setLearning(self):
        self._isLearning = True

    def unsetLearning(self):
        self._isLearning = False
        self._savedData = None

    def isLearning(self):
        return self._isLearning

    def saveData(self, data):
        if self._isLearning:
            self._savedData = data

    def getSavedData(self):
        return self._savedData

    def compute(self, tensor):
        raise NotImplementedError

    def learn(self, loss):
        raise NotImplementedError
