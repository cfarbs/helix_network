#!/usr/bin/env python

# Ensure python 3 forward compatibility
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T

class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''Layer of a nn, coputes s(Wx + b), where s is a nonlinearlity and x is input vector.
        :parameters:
            - W_init: np.ndarray, shape=(n_output, n_input)
                values to initialize the weight matrix to
            - b_init: np.ndarray, shape=(n_output,)
                values to initialize bias vector to
            - activation: theano.tensor.elemwise.Elemwise
                activation of function for layer output
        '''
        # retrieve the input and output dimensionally based on W's initiaization
        n_output, n_input = W_init.shape
        #make sure b is n_output in size
        assert b_init.shape == (n_output,)
        #all parameters should be shared variables.
        #used in the class to compute layer output
        #but are updated esewhere when optimizing network parameters
        #note that we are explicity requiring that W_init has the theano.config.floatX dtype
        self.W=theano.shared(value=W_init.astype(theano.config.floatX),
            #Name parameter for printing
            name='W',
            #borrow = true allows theano to use user memory for object
            #makes code sighty faster
            ##http://deepearning.net/software/theano/tutorial/aliasing.html
            borrow=True)
        #can force bias vector b to be column vector with numpy reshape method
        #when b is column vector, can pass matrix-shaped input to the layer
        #and get matrix shaped output, because broadcasting.
        self.b = theano.shared(value=b_init.reshape(n_output, 1).astype(theano.config.floatX),
            name='b',
            borrow=True,
            #Theano allows for broadcasting, similar to numpy.
            #however, must dneoate which aes can be broadcasted
            #By setting boradcastable=(False, True), denoting that b
            #can be broadcast (copied) along its 2nd dimension in order to be added
            #to another variable.
            #http://deeplearning.net/software/theano/library/tensor/basic.html
            broadcastable=(False,True))
        self.activation = activation
        #we will compute the gradient of cost of network with repesct to parameters in list
        self.params=[self.W,self.b]
    def output(self,x):
        '''
        Compute this layer's output given inputs
        :parameters:
            - x : theano.tensor.var.TensorVariable
                theano symbolic variable for layer inputs
        :returns:
            - output : theano.tensor.var.TensorVariable
                mixed, biased, and activated x
        '''
        #compute linear mix
        lin_output=T.dot(self.W,x) + self.b
        #output is just linear mix if no activation fxn
        #otherwise, apply the activation fxn
        return (lin_output if self.activation is None else self.activation(lin_output))

class MLP(object):
    def __init__(self, W_init, b_init, activations):
        '''
        Multi-layer perceptron class, computes composition of sequence of layers

        :parameters:
            - W_init : list of np.ndarray, len = N
                Values to initialize the weight matrix in each layer to.
                The layer sizees will be inferred from size of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.eemwise.Elemwise, len = N
                activation function for layer output of each layer
        '''
        #make sure input lists are all same length
        assert len(W_init) == len(b_init) == len(activations)
        #initialize list of layers
        self.layers= []
        #construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))
        #combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params
    def output (self, x):
        '''
        Compute the MLP's output given an input

        :parameters:
            - x : teano.tensor.var.TensorVariable
                Theano symbolic variable for network input
        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed thru MLP
        '''
        #recursively compute output
        for layer in self.layers:
            x=layer.output(x)
        return x
    def squared_error(self, x, y):
        '''
        Compute the squared euclidian error of the network output against the "true" output y

        :parameters:
                - x : theano.tensor.var.TensorVariable
                    theano symbolic variable for network input
                - y : theano.tensor.var.TensorVariable
                    theano symbolic variable for desired network output
        :returns:
                - error : theano.tensor.var.TensorVariable
                    The squared Euclidian distance between network output and y
        '''
        return T.sum((self.output(x)-y)**2)
    def gradient_updates_momentum(cost, params, learning_rate, momentum):
        '''
        Compute updates for gradient descent with momentum
        :parameters:
            - cost : theano.tensor.var.TensorVariable
                theano cost function to minimize
            - params: list of theano.tensor.TensorVariable
                    Parameters to compute gradient against
            - learning_rate : float
                gradient descent learning rate
            - momentum : float
                momentum parameter, should be at least 0 (standard gradient descent) and less than 1
        :returns:
            updates : list
                list of updates, one for each parameter
        '''
        #make sure momemtum is valid value
        assert momentum < 1 and momentum >= 0
        #list of update steps for each parameter
        updates = []
        #just gradient descent on cost
        for param in params:
                #for each parameter, create a param_update shared variable.
                #this variable will keep track of parameter's updaate across iterations.
                #we will initialize it to 0
                param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
                #each paramteter is updated by taking a step in the direction of the gradient
                #however, also "mix in" the previous step according to the given momentum value.
                #note that when updating param_update, using its old value and also the new gradient step.
                updates.append((param, param - learning_rate*param_update))
                #note that we don't need to derive backpropagation to compute updates - use T.grad instead!
                updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        return updates
