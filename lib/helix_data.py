#!/usr/bin/env python
import numpy as np
import os
import pickle
from random import shuffle
from .context_dict import contextdict
"""
Package to import helix data for use

"""
def load_helix(split,helixdict,adversarial = False,data):
    if adversarial:
        X = pickle.load(open("lib/helices.pkl","rb"))
        y = data
        print ("This is an adversarial run.")
    else:
        X = pickle.load(open("lib/helices.pkl","rb"))
        y = pickle.load(open("lib/randhelices.pkl","rb"))
        print ("This is a non-adversarial run.")

    aadict, features = contextdict(**helixdict)

    #print (aadict)
    Hsamples=[0]*len(X)
    nonHsamples=[1]*len(y)

    proX = []
    for num in range(len(X)):
        templist=[]
        tempobj = (X[num])
        for obs in range(len(tempobj)):
            templist.append((aadict[tempobj[obs]]))
            if len(templist)==len(tempobj):
                tempcomp = [item for sublist in templist for item in sublist]
                proX.append(tempcomp)
    proY = []
    for num in range(len(y)):
        templist=[]
        tempobj = (y[num])
        for obs in range(len(tempobj)):
            templist.append((aadict[tempobj[obs]]))
            if len(templist)==len(tempobj):
                tempcomp = [item for sublist in templist for item in sublist]
                proY.append(tempcomp)
    Xdata = list(zip(proX,Hsamples))
    Ydata = list(zip(proY,nonHsamples))
    dataset = Xdata + Ydata
    shuffle(dataset)
    split_point = int(len(Xdata) * split)
    xtrain_split = int(split_point + 0.5 * (1 - split) * len(Xdata))
    return dataset[:split_point], dataset[split_point:xtrain_split], dataset[xtrain_split:], features
