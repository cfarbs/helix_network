#!/usr/bin/env python
import numpy as np
import os
import pickle
from random import shuffle
from lib.context_dict import contextdict
"""
Package to import helix data for use

Data are retrieved from len=12_helices.pkl and len=12_rand.pkl


"""
def load_helix(split,helixdict):
    X = pickle.load(open("lib/gen_helices.pkl","rb"))
    y = pickle.load(open("lib/randhelices.pkl","rb"))

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
