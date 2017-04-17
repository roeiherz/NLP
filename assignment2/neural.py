#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive

def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    params[ofs:ofs+ Dx * H]
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Compute the probability
    ### YOUR CODE HERE: forward propagation

    # stage 1
    v1 = np.dot(data, W1) + b1
    z1 = sigmoid(v1)

    # stage 2
    v2 = np.dot(z1, W2) + b2
    z2 = softmax(v2)

    # add cost (cross entropy)
    probs = np.sum(z2 * label, axis=1)

    return probs
    ### END YOUR CODE

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    cost = 0
    # forward

    # stage 1
    v1 = np.dot(data, W1) + b1
    z1 = sigmoid(v1)

    # stage 2
    v2 = np.dot(z1, W2) + b2
    z2 = softmax(v2)

    # add cost (cross entropy)
    probs = np.sum(z2 * labels, axis=1)
    # Number of samples
    m = data.shape[0]
    # Calculate Cost
    cost = -np.sum(np.log(probs))

    # backward
    # According to section A: y^-1
    delta2 = z2 - labels    # [20, 10]
    gradW2 = np.dot(z1.T, delta2)
    gradb2 = np.dot(delta2.T, np.ones((m, 1))).T    # ([10, 20] dot [20, 1]) .T = [1, 10]
    # sigmoid grad z1 because the sigmoid_grad expects to get a sigmoid..
    delta1Out = np.dot(delta2, W2.T)  # [20, 5]
    delta1 = delta1Out * sigmoid_grad(z1)   # [20, 5]
    gradW1 = np.dot(data.T, delta1)    # [10, 5]
    gradb1 = np.dot(delta1.T, np.ones((m, 1))).T   # ([5, 20] dot [20, 1]) .T = [1, 5]

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
