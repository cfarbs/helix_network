#!usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T
import classes_example as C


#training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
np.random.seed(0)
#number of points
N = 1000
#labels for each cluster
y = np.random.random_integers(0,1,N)
#y = np.random.rand(3,2)
#mean of each cluster
means = np.array([[-1,1],[-1,1]])
#print (means)
#"""
#covariance (in X and y direction) of each cluster
covariances = np.random.random_sample((2,2))+1
print (covariances)

#dimensions of each point
X = np.vstack([np.random.randn(N)*covariances[0,y]+means[0,y],
np.random.randn(N)*covariances[1,y]+means[1,y]]).astype(theano.config.floatX)
#convert to targets, as floatX
y = y.astype(theano.config.floatX)
#plot the data
plt.figure(figsize=(8,8))
plt.scatter(X[0,:],X[1,:],c=y,lw=.3,s=3,cmap=plt.cm.cool)
plt.axis([-6,6,-6,6])
#plt.show()

#First, set the size of each layer (and number layers)
#input layer size is training data dimensionality (2)
#output size is just 1-d: class label - 0 or 1
#finally, let hidden layers be twice the size of input.
#if we wanted more layers, could add another layer size to the list
layer_sizes = [X.shape[0], X.shape[0]*2,1]
#set initial parameter values
W_init = []
b_init = []
activations = []
for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
    #getting the correct initialization matters a lot for non-toy problems.
    #however, here we can just use the following initialization with success:
    #normally distribute initial weights
    W_init.append(np.random.randn(n_output,n_input))
    #set initial biases to 1
    b_init.append(np.ones(n_output))
    #use sigmoid activation for all layers
    #note this doesn't make much sense when using squared distance
    #because sigmoid bounded on [0,1]
    activations.append(T.nnet.sigmoid)
#create instance of MLP class
mlp=C.MLP(W_init, b_init, activations)
#Create theano variables for the MLP input
mlp_input=T.matrix('mlp_input')
#and desired output
mlp_target=T.vector('mlp_target')
#learning rate and momentum hyperparameter values
#again, for non-toy values make a big difference
#as to whether network quickly converges on good local minimum
learning_rate=0.01
momentum=0.9
#create a fxn for computing the cost of the network given an input
cost = mlp.squared_error(mlp_input,mlp_target)
#create theano fxn for training network
train = theano.function([mlp_input, mlp_target], cost,
 updates=C.MLP.gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
#create a theano fxn for computing the MLP's output given some input
mlp_output = theano.function([mlp_input], mlp.output(mlp_input))
#keep track of the number of training iterations performed
iteration = 0
#only train network w/20 iterations
#more common technique is to use a hold-out validation set.
#when validation error starts to increase, the network is overfitting,
#so we stop training the net. This is early stopping; won't be done in this example.
max_iteration = 20
while iteration < max_iteration:
    #Train the network using entire training set.
    #With large datasets, its more common to use stochastic or mini-batch gradient descent
    #where only a subset (or single point) of training set is used at each iteration
    #this can also help network avoid loca minima.
    current_cost=train(X,y)
    #get current network output for all points in training set
    current_output = mlp_output(X)
    #can compute accuracy by threshoulding the output
    #and computing the proprtion of points whose class match the ground truth class.
    accuracy = np.mean((current_output > .5) == y)
    #plot network output after this iteration
    plt.figure(figsize=(8,8))
    plt.scatter(X[0,:],X[1,:],c=current_output,lw=.3,s=3,cmap=plt.cm.cool,vmin=0,vmax=1)
    plt.axis([-6,6,-6,6])
    plt.title('Cost: {:.3f}, Accuracy: {:.3f}'.format(float(current_cost), accuracy))
    plt.show()
    iteration+=1
