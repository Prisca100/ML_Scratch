import numpy as np

# Foward pass activation functions
# Sigmoid function
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = (Z)
    assert(A.shape == Z.shape)
    return A, cache

#Relu
def relu(Z):
    A = np.maximum(0,Z)
    cache = (Z)
    assert(A.shape == Z.shape)
    return A, cache

# Tanh 
def tanh(Z):
    A = np.tanh(Z)
    cache = (Z)
    assert(A.shape == Z.shape)
    return A, cache

# Backward Pass
# Sigmoid
def sigmoid_backward(dA, cache):
    
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

#ReLU
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    return dZ

# Shallow neural network
def layer_dims(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return(n_x, n_h, n_y)

