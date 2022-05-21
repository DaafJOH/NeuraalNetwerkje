import numpy as np
import math
#Preparation
i = np.matrix([
    [0, 0, 0], 
    [1, 0, 0], 
    [0, 1, 0],
    [1, 0, 1], 
    [1, 1, 1]])/3
t = np.matrix([
    [0], 
    [1], 
    [1], 
    [2], 
    [3]])/3
w1 = np.matrix([
    [0.1, 0.2, 0.7, 0.4, 0.5, 0.6], 
    [0.6, 0.5, 0.4, 0.9, 0.2, 0.1], 
    [0.4, 0.3, 0.5, 0.7, 0.9, 0.3]])
w2 = np.matrix([[0.5], 
    [0.4], 
    [0.8], 
    [0.8], 
    [0.3],
    [0.2]])
lr = 0.007
# Functions
def sigmoid(x):
    return 1/(1+np.exp(x))
def dsigmoid(x):
    return np.multiply(sigmoid(x),(1-sigmoid(x)))
# Loop
for k in range(1000):
    # Forward
    x1 = i * w1
    y1 = sigmoid(x1)

    x2 = y1 * w2
    y2 = sigmoid(x2)
    
    #Backward
    deltaN = np.multiply((y2-t),np.multiply(y2,(1-y2)))
    deltan = np.multiply(deltaN, dsigmoid(x2))*w2.transpose()
    w2 = w2 - lr * (deltaN.transpose()*y1).transpose()
    w1 = w1 - lr * (i.transpose()*np.multiply(deltan, dsigmoid(x1)))
# Stop Loop
print(y2*100)