#!usr/bin/env python
from __future__ import print_function
import numpy as np
import unittest
from lib.optimization import mini_batch_sgd
from lib.helix_data import load_helix

class HelixNetwork(unittest.TestCase):
    def setUp(self):
        tr_data, xtr_data, ts_data = load_helix(0.7)
        self.tr = np.array([x[0] for x in tr_data])
        self.tr_l = [x[1] for x in tr_data]
        self.xtr = np.array([x[0] for x in xtr_data])
        self.xtr_l = [x[1] for x in xtr_data]
        self.ts = np.array([x[0] for x in ts_data])
        self.ts_l = [x[1] for x in ts_data]
    def CheckHelix(self, test_name, model_type, hidden_dim, verbose, epochs, batch_size=10, extra_args=None):
        net, results = mini_batch_sgd(motif=test_name,
                                      train_data=self.tr, labels=self.tr_l,
                                      xTrain_data=self.xtr, xTrain_targets=self.xtr_l,
                                      learning_rate=0.001, L1_reg=0.0, L2_reg=0.0, epochs=epochs,
                                      batch_size=batch_size, hidden_dim=hidden_dim, model_type=model_type,
                                      model_file=None, trained_model_dir=None, verbose=verbose, extra_args=extra_args)
        #self.assertTrue(results['batch_costs'][1] > results['batch_costs'][-1])
        #self.assertTrue(results['xtrain_accuracies'][1] < results['xtrain_accuracies'][-1])
    def StartingNN(self):
        self.CheckHelix(test_name="twoLayerTest", model_type="twoLayer", hidden_dim=[10], verbose=True, epochs=1000)

def main():
    testSuite = unittest.TestSuite()
    testSuite.addTest(HelixNetwork('StartingNN'))
    testRunner = unittest.TextTestRunner(verbosity=2)
    testRunner.run(testSuite)


if __name__ == '__main__':
    main()
