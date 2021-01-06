import matplotlib.pyplot as plt
import numpy as np
import math
import random

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_derivative(x):
    return (x)*(1.0 - (x))
    #would ususally be written as sigmoid(x) here but
    #the current function approximates much faster (~ 20x faster)

class NeuralNetwork:
    def __init__(self,x,y, n):
        self.input = x
        #Initialising weights
        #Here the n refers to the number of nodes in the hidden layer
        self.weights1 = np.random.rand(self.input.shape[1],n)
        self.weights2 = np.random.rand(n,1)
        self.y =y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        #Working out the adjustments to be made
        d_weights2 = np.dot(self.layer1.T, (2*(self.y -self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        #applying the adjustments
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def pushforward(self, X):
        #Function designed for the completed network feeding in 
        #testing data
        self.input = X
        nn.feedforward()
        return self.output
    
if __name__ == "__main__":
    #Training data left as 1 decimal place for brevity
    X = np.array([[0.4,0.5],
                [0.5,0.6],
                [0.6,0.4],
                [0.7,0.6],
                [0.9,0.2],
                [0.1,0.5],
                [0.1,0.2],
                [0.4,0.3],
                [0.3,0.5],
                [0.6,0.9]])
    #Here we have a coordinate between 0 and 1
    #Labelling the corresponding y value to determing if it is in
    #Category A or Category B
    y = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

    n = 8 
    #n here represents the number of neurons in the hidden layer
    nn = NeuralNetwork(X,y, n)


    n_iterations = 3000
    #number of iterations, tends to become accurate on training data
    #for n> 1000
    for i in range(n_iterations):
        nn.feedforward()
        nn.backprop()

    #To be used for checking if the output of the training data 
    #is accurate
    #print(nn.output)

    #Will need to be atleast a multiple of 10 so that our training data
    #Corresponds to a point, this determines resolution of image
    resolution = 100

    #Creates an array of zeros that we will use for creating the image
    img = np.zeros((resolution,resolution))

    #Used for determing colour gradient at each point
    for i in range(resolution):
        for j in range(resolution):
            coordinate = np.reshape(np.array([i/resolution, j/resolution]), (1,2))
            #Can alternate here for gradient view or strict view
            #img[i][j] = nn.pushforward(coordinate) * 255
            img[i][j] = np.around(nn.pushforward(coordinate), decimals =0) * 255
    
    #This is for displaying the training data on the graph
    for i in range(X.shape[0]):
        if(y[i] == 0):
            img[int(X[i,0]*resolution)][int(X[i,1]*resolution)] = -1* 255
        else:
            img[int(X[i,0]*resolution)][int(X[i,1]*resolution)] = 2* 255

    #Sets the colour map to red/blue    
    cmap = plt.get_cmap('RdBu')
    #Here we use this so we can separate the sets into
    #white and black points
    cmap.set_under('white')
    cmap.set_over('black')

    #vmin and vmax slightly altered to allow for upper and lower bound cases
    plt.imshow(img, cmap=cmap, vmin=-1, vmax=256)
    plt.show()
