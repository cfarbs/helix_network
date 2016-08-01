#!usr/bin/env python

from __future__ import print_function
import os
import theano
import sys
import glob
import pickle as cPickle
import pandas as pd
import numpy as np
import theano.tensor as T
from itertools import chain
from .model import NeuralNetwork
from random import shuffle
from .helix_data import load_helix

def setUp(split,helixdict,data,adversarial):
    tr_data,xtr_data,ts_data, features = load_helix(split,helixdict,data,adversarial)
    train_data = np.array([x[0] for x in tr_data])
    labels = [x[1] for x in tr_data]
    xTrain_data = np.array([x[0] for x in xtr_data])
    xTrain_targets = [x[1] for x in xtr_data]
    test_data = np.array([x[0] for x in ts_data])
    test_targets = [x[1] for x in ts_data]
    return train_data, labels, xTrain_data, xTrain_targets, test_data, test_targets, features

def preprocess_data(training_vectors, xtrain_vectors, test_vectors, preprocess=None):
    assert(len(training_vectors.shape) == 2 and len(xtrain_vectors.shape) == 2 and len(test_vectors.shape) == 2)
    if preprocess == "center" or preprocess == "normalize":
        training_mean_vector = np.nanmean(training_vectors, axis=0)
        training_vectors -= training_mean_vector
        xtrain_vectors -= training_mean_vector
        test_vectors -= training_mean_vector

        if preprocess == "normalize":
            training_std_vector = np.nanstd(training_vectors, axis=0)
            training_vectors /= training_std_vector
            xtrain_vectors /= training_std_vector
            test_vectors /= training_std_vector

    prc_training_vectors = np.nan_to_num(training_vectors)
    prc_xtrain_vectors = np.nan_to_num(xtrain_vectors)
    prc_test_vectors = np.nan_to_num(test_vectors)

    return prc_training_vectors, prc_xtrain_vectors, prc_test_vectors


def get_network(x, in_dim, n_classes, hidden_dim, model_type, extra_args=None):
    if model_type == "twoLayer":
        return NeuralNetwork(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)
    if model_type == "threeLayer":
        return ThreeLayerNetwork(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)
    if model_type == "ReLUthreeLayer":
        return ReLUThreeLayerNetwork(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)
    if model_type == "fourLayer":
        return FourLayerNetwork(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)
    if model_type == "ReLUfourLayer":
        return FourLayerReLUNetwork(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim)
    if model_type == "ConvNet3":
        return ConvolutionalNetwork3(x=x, in_dim=in_dim, n_classes=n_classes, hidden_dim=hidden_dim,
                                     **extra_args)
    else:
        print("Invalid model type", file=sys.stderr)
        return False


def find_model_path(model_directory, title):
    p = "{modelDir}/{title}_Models/summary_stats.pkl".format(modelDir=model_directory, title=title)
    print("p={}".format(p))
    assert(os.path.exists(p)), "didn't find model files in this directory"
    summary = cPickle.load(open(p, 'rb'))
    assert('best_model' in summary), "summary file didn't have the best_model file path"
    model = summary['best_model'].split("/")[-1]  # disregard the file path
    print("model: {}".format(model))
    path_to_model = "{modelDir}/{title}_Models/{model}".format(modelDir=model_directory, model=model, title=title)
    print("loading model from {}".format(path_to_model))
    return path_to_model


def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """

        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
