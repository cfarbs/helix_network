#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
#infile = input("enter the set of features from the combo tests to examine")
#infilestr=str(infile)
#print (type(infilestr))

epochs = np.arange(0,2500,250)
#savedir = "combo_train_graphs/"
#data = pickle.load(open("combo_80.pkl","rb"))
#for info in data:
"""results = pickle.load(open("gen_rand_helices_2_0379_Models/summary_stats.pkl","rb"))
train_accuracies = (results['train_accuracies'])
xtrain_accuracies = (results['xtrain_accuracies'])
plt.figure()
plt.plot(epochs, results['train_accuracies'])
plt.plot(epochs, results['xtrain_accuracies'])
plt.ylabel("Accuracy of Train/Xtrain")
plt.xlabel("Epochs")
plt.title("Accuracy Plot for 0379 on Generated vs Random; 2nd Run")
plt.legend(['Training Accuracies', 'Xtrain Accuracies'], loc = 'lower right')
plt.savefig("generated_data_plots/0379_gen_rand_2.png")
percents = []"""
percents = []
infile = open("generated_helices-nolearn_0379_Models/statsummary.txt","r")

for line in infile:
    lines = line.split(",")
    try:
        point = lines[7]
        digit = point.strip("\n")
        #print (digit)
        try:
            numb = float(digit)
            percents.append(numb)
        except:
            pass
    except:
        pass
#for line in infile:
#    digit = line.strip("\n")
#    numb = float(digit)
#    percents.append(numb)
(mu, sigma)=norm.fit(percents)
n, bins, patches = plt.hist(percents, 10, normed =1, facecolor='blue',alpha =0.75)
y = mlab.normpdf(bins,mu,sigma)
l=plt.plot(bins,y,'r--',linewidth=2)
plt.xlabel("Test Accuracy")
plt.ylabel("Frequency")
#figure_title = "0379 on Generated and Random Data; 2nd Run, n = 100 iterations"
plt.title(r'$\mathrm{Histogram\ of\ Errors\ for\ 0379\ for\ Real\ and\ Generated\ data:}\ \mu=%.3f,\ \sigma=%.3f$'%(mu,sigma), y = 1.04)
plt.show()
#plt.savefig("generated_data_plots/Error_dist_Gen_Rand_0379_2.png")

#infile.close()
"""
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
plt.savefig(savedir+infilestr+".png")
#fileloc=input("Enter the name of the model directory to analyze")
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
"""

#infilename = "testno_feats_Models/test_probs.pkl"
#data= pickle.load(open(infilename,"rb"))
#print (data[0])
"""
truepos = 0
trueneg = 0
falsepos = 0
falseneg = 0
for stuff in data:
    if stuff[1] == 0:
        if stuff[2][0] > stuff[2][1]:
            truepos +=1
        else:
            falseneg += 1
    elif stuff[1] == 1:
        if stuff[2][1] > stuff [2][0]:
            trueneg += 1
        else:
            falsepos +=1
    else:
        print (stuff[1])
        print ("Some how, a third value was decided on by a binary classifier. Weird, huh?")

#print ("True Positives:", truepos)
#print ("False Negatives:", falseneg)
#print ("True Negatives:", trueneg)
#print ("False Positives:", falsepos)

tpr = truepos/(truepos+falseneg)
fpr = falsepos/(falsepos+trueneg)

print (tpr)
print (fpr)
"""

"""x = []
y = []
for points in data:
    x.append(points[0])
    y.append(points[1])
#x = data[0]
#y = data[1]
print (x[0],y[0])
plt.plot(x,y)
plt.show()
"""
