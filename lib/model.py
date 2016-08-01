#!/usr/bin/env python
"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""
from __future__ import print_function
import sys
import pickle as cPickle
import numpy as np
izip = zip
#from itertools import izip
from .layers import HiddenLayer, SoftmaxLayer
import theano.tensor as T

class Model(object):
    """Base class for network models
    """
    def __init__(self, x, in_dim, n_classes, hidden_dim):
        """x: input data
          in_dim: dimensionality of input data
          n_classes: number of classes within data
        """
        self.input = x
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.params = None
        self.initialized = False

    def write(self, file_path):
        """Write model to file, using cPickle
        file_path: string, path to and including file to be written
        """
        f = open(file_path, 'wb')
        d = {
            "model": self.__class__,
            "in_dim": self.in_dim,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
        }
        assert (self.params is not None)
        for param in self.params:
            lb = '{}'.format(param)
            d[lb] = param.get_value()
        cPickle.dump(d, f)

    def load_from_file(self, file_path=None, model_obj=None, careful=False):
        """Load model
         file_path: string, file to and including model file
        """
        if file_path is not None:
            f = open(file_path, 'rb')
            d = cPickle.load(f)
        else:
            assert(model_obj is not None), "need to provide file or dict with model params"
            d = model_obj

        assert(self.in_dim == d['in_dim'])
        assert(self.n_classes == d['n_classes']), "Incorrect number of input classes, got {0} should be {1}".format(
            d['n_classes'], self.n_classes)
        assert(self.__class__ == d['model'])
        assert(self.hidden_dim == d['hidden_dim'])

        missing_params = 0
        for param in self.params:
            look_up = "{}".format(param)
            if look_up in d.keys():
                assert(len(param.get_value()) == len(d[look_up]))
                param.set_value(d[look_up])
            else:
                missing_params += 1
        if careful is True:
            print("got {} missing params".format(missing_params), file=sys.stderr)

    def load_from_object(self, model, careful=False):
        self.load_from_file(file_path=None, model_obj=model, careful=careful)


class NeuralNetwork(Model):
    def __init__(self, x, in_dim, hidden_dim, n_classes):
        assert(len(hidden_dim) == 1)
        super(NeuralNetwork, self).__init__(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)

        self.input = x

        # first layer (hidden)
        self.hidden_layer0 = HiddenLayer(x=x, in_dim=in_dim, out_dim=hidden_dim[0], layer_id='h1', activation=T.tanh)

        # final layer (softmax)
        self.softmax_layer = SoftmaxLayer(x=self.hidden_layer0.output, in_dim=hidden_dim[0], out_dim=n_classes,
                                          layer_id='s0')

        # Regularization
        self.L1 = abs(self.hidden_layer0.weights).sum() + abs(self.softmax_layer.weights).sum()
        self.L2_sq = (self.hidden_layer0.weights ** 2).sum() + (self.softmax_layer.weights ** 2).sum()

        # output, errors, and likelihood
        self.y_predict = self.softmax_layer.y_predict
        self.negative_log_likelihood = self.softmax_layer.negative_log_likelihood
        self.errors = self.softmax_layer.errors
        self.output = self.softmax_layer.output
        self.params = self.hidden_layer0.params + self.softmax_layer.params


        # network type
        self.type = "twoLayer"
