from math import exp

class Network:
    def __init__(self, startingLayer: int, *hiddenLayers: int): # The function that run when the Network object is created
        self.layers = [] # Make a variable that contains a matrix of all the layers and neurons
        self.startingLayer = startingLayer # Remember the number of first layer neurons
        for layerIndex, layer in enumerate(hiddenLayers): # Cycle through all layers
            self.layers.append([])
            for neuron in range(layer):
                if layerIndex == 0: self.layers[layerIndex].append(Neuron(startingLayer))
                else: self.layers[layerIndex].append(Neuron(hiddenLayers[layerIndex-1]))

    def Calculate(self, input: list) -> list:
        if len(input) != self.startingLayer: raise TypeError("length of input must be equal to staring neurons")
        for i, v in enumerate(input, 0):
            for j, w in enumerate(self.layers[0], 0):
                w.value += w.weights[i] * v
        for i, v in enumerate(self.layers, 0):
            for j, w in enumerate(v, 0):
                w.CalcActivation()
                try:
                    for k, x in enumerate(self.layers[i+1]):
                        x.value += x.weights[j] + w.activation
                except IndexError: pass
        output = []
        for i in self.layers[-1]:
            output.append(i.activation)
        return output


class Neuron:
    def __init__(self, neuronCount):
        self.bias = 0
        self.activation = 0
        self.value = 0
        self.weights = [1 for i in range(0, neuronCount)]

    def CalcActivation(self):
        self.activation = sigmoid(self.value + self.bias)

def sigmoid(x: int) -> int:
    return 1 / (1 + exp(-x))