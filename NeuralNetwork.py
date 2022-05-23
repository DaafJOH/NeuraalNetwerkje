from math import exp

class Network:
    def __init__(self, startingLayer: int, *hiddenLayers: int): # The function that run when the Network object is created
        self.layers = [] # Make a variable that contains a matrix of all the layers and neurons
        self.startingLayer = startingLayer # Remember the number of first layer neurons
        for layerIndex, layer in enumerate(hiddenLayers): # Cycle through all layers except the first
            self.layers.append([]) # Make a empty list for the layer
            for neuron in range(layer): # Cycle through all neurons in the layer
                if layerIndex == 0: self.layers[layerIndex].append(Neuron(startingLayer)) # If it is the first hidden layer make a neuron with [startingLayer] amount of weights
                else: self.layers[layerIndex].append(Neuron(hiddenLayers[layerIndex-1])) # If not, make a neuron with the amount of wheight as the amount of neurons in the previous layer 

    def Calculate(self, inputList: list) -> list: # Function to calculate what the network would do with a certain input
        if len(inputList) != self.startingLayer: raise TypeError("length of input must be equal to staring neurons") # If the number of inputs are wrong raise error
        for inputIndex, inputNumber in enumerate(inputList): # Cycle through all input numbers
            for neuron in self.layers[0]: # Cycle through all first hidden layer neurons
                neuron.value += neuron.weights[inputIndex] * inputNumber # Calculate the value of the connection
        for layerIndex, layer in enumerate(self.layers): # Cycle through all layers except the first
            for neuronIndex, neuron in enumerate(layer): # Cycle through all neurons in the layer
                neuron.CalcActivation() # Calculate the activation of the neuron
                try: 
                    for nextNeuron in self.layers[layerIndex+1]: # Cycle through all the neurons in the next layer
                        nextNeuron.value += nextNeuron.weights[neuronIndex] * neuron.activation # Calculate the value of the connection
                except IndexError: continue # If a IndexError is caught (if the output layer has been reached) continue
        output = [] # Make a output variable
        for i in self.layers[-1]: # Cycle through the output layer
            output.append(i.activation) # Put all the output values in the output list
        return output # Return the output

    def Choice(self, inputList: list) -> int: # A function that returns the index of the firt highest output neuron
        output = self.Calculate(inputList)
        return output.index(max(output))


class Neuron: 
    def __init__(self, neuronCount): # The function that run when the Neuron object is created
        self.bias = 0 # Variable for the bias
        self.activation = 0 # Variable for the activation
        self.value = 0 # Variable for the uncalculated activation
        self.weights = [1 for i in range(0, neuronCount)] # List of all the wheight

    def CalcActivation(self): # Function to calculate the activation
        self.activation = sigmoid(self.value + self.bias)

def sigmoid(x: float) -> float: return 1 / (1 + exp(-x))