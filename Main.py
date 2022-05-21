import NeuralNetwork
network = NeuralNetwork.Network(784, 16, 16, 10)
print(network.Calculate([1 for i in range(0, 784)]))
print(network.Choice([1 for i in range(0, 784)]))