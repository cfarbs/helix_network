#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


# globals
RNG = np.random.RandomState()


class SoftmaxLayer(object):
    def __init__(self, x, in_dim, out_dim, layer_id):
        self.weights = theano.shared(value=np.zeros([in_dim, out_dim], dtype=theano.config.floatX),
                                     name=layer_id + 'weights',
                                     borrow=True
                                     )
        self.biases = theano.shared(value=np.zeros([out_dim], dtype=theano.config.floatX),
                                    name=layer_id + 'biases',
                                    borrow=True
                                    )
        self.params = [self.weights, self.biases]

        self.input = x
        self.output = self.prob_y_given_x(x)
        # maybe put a switch here to check for nan/equivalent probs
        self.y_predict = T.argmax(self.output, axis=1)

    def prob_y_given_x(self, input_data):
        return T.nnet.softmax(T.dot(input_data, self.weights) + self.biases)

    def negative_log_likelihood(self, labels):
        return -T.mean(T.log(self.output)[T.arange(labels.shape[0]), labels])

    def errors(self, labels):
        return T.mean(T.neq(self.y_predict, labels))

class HiddenLayer(object):
    def __init__(self, x, in_dim, out_dim, layer_id, W=None, b=None, activation=T.tanh):
        if W is None:
            W_values = np.asarray(RNG.uniform(low=-np.sqrt(6. / (in_dim + out_dim)),
                                              high=np.sqrt(6. / (in_dim + out_dim)),
                                              size=(int(in_dim), int(out_dim))),
                                  dtype=theano.config.floatX
                                  )
            W = theano.shared(value=W_values, name=layer_id + 'weights', borrow=True)
        if b is None:
            b_values = np.zeros((out_dim,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=layer_id + 'biases', borrow=True)

        self.weights = W
        self.biases = b
        self.params = [self.weights, self.biases]

        self.input = x
        lin_out = T.dot(x, self.weights) + self.biases
        self.output = lin_out if activation is None else activation(lin_out)



"""class Layer(object):
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
"""
