#!/usr/bin/env python

"""
Citations
[1]http://cs231n.github.io/neural-networks-case-study/
[2]http://nbviewer.ipython.org/github/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
[3]https://github.com/mnielsen/neural-networks-and-deep-learning
"""

from __future__ import print_function
import sys, os
import theano
import theano.tensor as T
import numpy as np
import pickle as cPickle
from itertools import chain
from lib.helix_utils import setUp, preprocess_data, get_network,find_model_path
from lib.optimization import mini_batch_sgd

def predict(test_data, true_labels, batch_size, model, model_file=None):
    if model_file is not None:
        print("loading model from {}".format(model_file), end='\n', file=sys.stderr)
        model.load_from_file(file_path=model_file, careful=True)
    n_test_batches = test_data.shape[0] / batch_size
    y = T.ivector('y')

    prob_fcn = theano.function(inputs=[model.input],
                               outputs=model.output,
                               allow_input_downcast=True)

    error_fcn = theano.function(inputs=[model.input, y],
                                outputs=model.errors(y),
                                allow_input_downcast=True)

    errors = [error_fcn(test_data[x * batch_size: (x + 1) * batch_size],
                        true_labels[x * batch_size: (x + 1) * batch_size])
              for x in range(int(n_test_batches))]

    probs = [prob_fcn(test_data[x * batch_size: (x + 1) * batch_size])
             for x in range(int(n_test_batches))]

    probs = list(chain(*probs))
    probs = list(zip(test_data, true_labels, probs))
    print (len(errors))
    return errors, probs


def evaluate_network(test_data, targets, model_file, model_type, batch_size, extra_args=None):
    # load the model file
    model = cPickle.load(open(model_file, 'r'))
    n_train_samples, data_dim = test_data.shape
    n_classes = len(set(targets))
    if data_dim != model['in_dim'] or n_classes != model['n_classes']:
        print("This data is not compatible with this network, exiting", file=sys.stderr)
        return False
    net = get_network(x=test_data, in_dim=model['in_dim'], n_classes=model['n_classes'], model_type=model_type,
                      hidden_dim=model['hidden_dim'], extra_args=extra_args)
    net.load_from_object(model=model, careful=True)
    errors, probs = predict(test_data=test_data, true_labels=targets, batch_size=batch_size, model=net, model_file=None)
    return errors, probs


def classify_with_network2(
        # alignment files
        preprocess, title, helixdict,
        # training params
        learning_algorithm, train_test_split, iterations, epochs, batch_size,
        # model params
        learning_rate, L1_reg, L2_reg, hidden_dim, model_type, model_dir=None, extra_args=None,
        # output params
        out_path="./"):
    print("2 way classification")
    out_file = open(title+"_summary.txt", 'a')
    if model_dir is not None:
        print("looking for model in {}".format(model_dir))
        model_file = find_model_path(model_dir, title)
    else:
        model_file = None

    split = train_test_split

    training_data, training_labels, xtrain_data, xtrain_targets, test_data, test_targets, features = setUp(split,helixdict)

    for i in range(iterations):
        prc_train, prc_xtrain, prc_test = preprocess_data(training_vectors=training_data,
                                                          xtrain_vectors=xtrain_data,
                                                          test_vectors=test_data,
                                                          preprocess=preprocess)

        # bin to hold accuracies for each iteration
        scores = []

        collect_data_vectors_args = {
            "portion": train_test_split,
        }

        # evaluate

        trained_model_dir = "{0}{1}_Models/".format(out_path, title)

        training_routine_args = {
            "motif": title,
            "train_data": training_data,
            "labels": training_labels,
            "xTrain_data": xtrain_data,
            "xTrain_targets": xtrain_targets,
            "learning_rate": learning_rate,
            "L1_reg": L1_reg,
            "L2_reg": L2_reg,
            "epochs": epochs,
            "batch_size": batch_size,
            "features": features,
            "hidden_dim": hidden_dim,
            "model_type": model_type,
            "model_file": model_file,
            "trained_model_dir": trained_model_dir,
            "extra_args": extra_args
        }

        #print (type(training_routine_args['trained_model_dir']))
        net, summary = mini_batch_sgd(**training_routine_args)

        errors, probs = predict(prc_test, test_targets, training_routine_args['batch_size'], net,
                                model_file=summary['best_model'])
        errors = 1 - np.mean(errors)
        owtfile = "{}statsummary.txt".format(trained_model_dir)
        print("{0}: {1} test accuracy.".format(title, (errors * 100)))
        out_file.write("{0}: {1} test accuracy.\n".format(title, (errors * 100)))
        outfile2 = open(owtfile, "a")
        outfile2.write("{0}\n".format((errors * 100)))
        scores.append(errors)
        outfile2.close
        with open("{}test_probs.pkl".format(trained_model_dir), 'wb') as probs_file:
            cPickle.dump(probs, probs_file)

    print(">{motif}\t{accuracy}".format(motif=title, accuracy=np.mean(scores), end="\n"), file=out_file)
    return net
