from math import exp
from numpy import subtract, floor

class Network:
    def __init__(self, startingLayer: int, *hiddenLayers: int):  # The function that run when the Network object is created
        self.layers = []  # Make a variable that contains a matrix of all the layers and neurons
        self.startingLayer = startingLayer  # Remember the number of first layer neurons
        for layerIndex, layer in enumerate(hiddenLayers):  # Cycle through all layers except the first
            self.layers.append([])  # Make an empty list for the layer
            for neuron in range(layer):  # Cycle through all neurons in the layer
                if layerIndex == 0:
                    self.layers[layerIndex].append(Neuron(startingLayer))  # If it is the first hidden layer make a neuron with [startingLayer] amount of weights
                else:
                    self.layers[layerIndex].append(Neuron(hiddenLayers[layerIndex - 1]))  # If not, make a neuron with the amount of weight as the amount of neurons in the previous layer

    def Calculate(self, inputList: list) -> list:  # Function to calculate what the network would do with a certain input
        if len(inputList) != self.startingLayer: raise TypeError(
            "length of input must be equal to staring neurons")  # If the number of inputs are wrong raise error
        for inputIndex, inputNumber in enumerate(inputList):  # Cycle through all input numbers
            for neuron in self.layers[0]:  # Cycle through all first hidden layer neurons
                neuron.value += neuron.weights[inputIndex] * inputNumber  # Calculate the value of the connection
        for layerIndex, layer in enumerate(self.layers):  # Cycle through all layers except the first
            for neuronIndex, neuron in enumerate(layer):  # Cycle through all neurons in the layer
                neuron.CalcActivation()  # Calculate the activation of the neuron
                try:
                    for nextNeuron in self.layers[layerIndex + 1]:  # Cycle through all the neurons in the next layer
                        nextNeuron.value += nextNeuron.weights[neuronIndex] * neuron.activation  # Calculate the value of the connection
                except IndexError:
                    continue  # If a IndexError is caught (if the output layer has been reached) continue
        output = []  # Make a output variable
        for i in self.layers[-1]:  # Cycle through the output layer
            output.append(i.activation)  # Put all the output values in the output list
        return output  # Return the output

    def Choice(self, inputList: list) -> int:  # A function that returns the index of the first highest output neuron
        output = self.Calculate(inputList)
        return output.index(max(output))

    def Cost(self, data:list, target:list):  # A function to calculate cost
        output = self.Calculate(data)
        cost = []
        for d, i in enumerate(output):
            if d < target[i]: cost.append(d-target[i] ** 2)
            else: cost.append(0-(d-target[i]) ** 2)
        return cost

    def Train(self, data:list, labels:list, miniBatchSize:int):  # A function to train network
        for i in range(floor(len(data)/miniBatchSize)):
            miniBatch = data[i*miniBatchSize:i*miniBatchSize+miniBatchSize]
            miniBatchL = labels[i * miniBatchSize:i * miniBatchSize + miniBatchSize]
            for example, index in enumerate(miniBatch):
                for cost in self.Cost(example, miniBatchL[index])

class Neuron:
    def __init__(self, neuronCount):  # The function that run when the Neuron object is created
        self.bias = 0  # Variable for the bias
        self.activation = 0  # Variable for the activation
        self.value = 0  # Variable for the uncalculated activation
        self.weights = [1 for i in range(0, neuronCount)]  # List of all the weight

    def CalcActivation(self):  # Function to calculate the activation
        self.activation = sigmoid(self.value + self.bias)


def sigmoid(x: float) -> float: return 1 / (1 + exp(-x))
