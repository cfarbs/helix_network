#!/usr/bin/env python

import pickle
import os
import numpy as np
from itertools import combinations
from random import sample
from lib.context_dict import contextdict

###################################################
j = {
    "experiment_name": "default_random",
    "hidden_dim": [10],
    "model_type": "twoLayer",
    "helixdict" : []
}
for m in range(10):
    d = dict()
    d['title'] = "default_random"
    d['choose'] = None
    d['rand'] = True
    d['args'] = False
    j['helixdict'].append(d)
pickle.dump(j,open("config/defaultrandom.pkl",'wb'))

###################################################
j = {
    "experiment_name": "All combinations",
    "hidden_dim": [20],
    "model_type": "twoLayer",
    "helixdict": []
}
choices = combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],4)
for m in choices:
    m_list = list(m)
    d = dict()
    d['title'] = str(m)
    d['choose'] = m_list
    d['rand'] = False
    d['args'] = True
    j['helixdict'].append(d)
pickle.dump(j,open("config/combos.pkl",'wb'))


###################################################
