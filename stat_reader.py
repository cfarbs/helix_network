#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt

#infile = input("enter the set of features from the combo tests to examine")
#infilestr=str(infile)
#print (type(infilestr))

epochs = np.arange(0,5000,500)
#savedir = "combo_train_graphs/"
#data = pickle.load(open("combo_80.pkl","rb"))
"""for info in data:
    infilestr = info[0]
    acc = str(info[1])
    results = pickle.load(open("combo_models/"+infilestr+"_Models/summary_stats.pkl","rb"))
    error = pickle.load(open("combo_models/"+infilestr+"_Models/test_probs.pkl","rb"))
    train_accuracies = (results['train_accuracies'])
    xtrain_accuracies = (results['xtrain_accuracies'])
    plt.figure()
    plt.plot(epochs, results['train_accuracies'])
    plt.plot(epochs, results['xtrain_accuracies'])
    plt.ylabel("Accuracy of Train/Xtrain")
    plt.xlabel("Epochs")
    plt.title("Accuracy Plot for "+infilestr+" "+"Test Accuracy: "+acc)
    plt.legend(['Training Accuracies', 'Xtrain Accuracies'], loc = 'lower right')
    plt.savefig(savedir+infilestr+".png")"""
infilename = "3079_summary.txt"
infile = open(infilename,"r")
names = []
percents = []

for line in infile:
    lines = line.split(":")
    digit = lines[1].split(" ")
    numb = float(digit[1])
    names.append(lines[0])
    percents.append(numb)

plt.xlabel("Test Accuracy")
plt.ylabel("Frequency")
plt.title("Error distribution for 0379, n = 100 iterations")
plt.hist(percents)
plt.show()



#data=list(zip(names,percents))




#feats = (results["features"])
#bestmodel =  (results['best_model'])

#xtrain_errors = (results["xtrain_errors"])


#outfile = "testdefault_random_Models/statsummary.txt"
#owtfile = open(outfile, "a")
#outstatement = ("{0},{1},{2},{3},{4}\n".format(feats, bestmodel, train_accuracies, xtrain_accuracies,xtrain_errors))
#owtfile.write(outstatement)
#owtfile.close()

#Order of data in file:
#test accuracy,feature1,feature2,feature3,feature4,bestmodel,
#training accuracy, xtrain accuracy, xtrain error
#
