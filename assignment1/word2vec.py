#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from gradcheck import gradcheck_naive
from sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    if len(x.shape) > 1:
        # Implementation Matrix

        # Calc L2 norm
        l2_distance_rows = np.sqrt((x * x).sum(axis=1))
        new_matrix = (x.transpose() / l2_distance_rows[:]).transpose()
        return new_matrix

    else:
        # Implementation a row Vector

        max_num = np.sqrt((x * x).sum(axis=0))
        new_matrix = x / max_num
        return new_matrix



def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print x
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
                 [1 x D]
    target -- integer, the index of the target word [1]
    outputVectors -- "output" vectors (as rows) for all tokens [N X D]
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """


    # Define Uo, Vc
    Uo = outputVectors[target]  # [1 X D]
    Vc = predicted # [1 X D]

    # cost
    X = np.dot(predicted, outputVectors.transpose()) # [1 X N]
    softmax_vec = softmax(X) # [1 X N]
    log_softmax = np.log(softmax_vec)
    cost = - log_softmax[target]

    # grad (Vc)
    gradPred = - Uo + np.dot(softmax_vec, outputVectors) # [1 X D]

    # grad (U)
    grad =  np.dot(softmax_vec.reshape((-1, 1)), Vc.reshape((-1, 1)).T) # [N X D]
    grad[target] -= Vc

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.
       `
    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    # Define Uo, Vc
    Uo = outputVectors[target]
    Vc = predicted

    ## Cost
    # Define x = Uo.Vc
    pos_x = np.dot(Uo, Vc.transpose()) # [scalar]

    # Sigmoid
    sigma = sigmoid(pos_x) # [scalar]

    # positive J of the cost function
    pos_J = -np.log(sigma) # [scalar]
    ## we could implement more efficent code - but we decided to seperate the calculation of the cost and the grads
    ## for clearer implementation
    # calc neg_J
    neg_J = 0
    for i in range(1, K + 1):
        Uo_neg = outputVectors[indices[i]]
        neg_x = -1 * np.dot(Uo_neg, Vc.transpose()) # [scalar]
        neg_J += -np.log(sigmoid(neg_x)) # [scalar]
    cost = pos_J + neg_J # [scalar]

    ## Grad (Vc)
    grad_pos = (sigma - 1) * Uo # [1 X D]
    grad_neg = 0
    for i in range(1, K + 1):
        Uo_neg = outputVectors[indices[i]]
        neg_x = -1 * np.dot(Uo_neg, Vc.transpose()) # scalar
        grad_neg += (sigmoid(neg_x) - 1) * Uo_neg # [1 X D]
    gradPred = grad_pos - grad_neg # [1 X D]

    ## gradpred (Uo)
    grad = np.zeros(outputVectors.shape) # [N X D]
    grad[target] = (sigma - 1) * Vc # [ 1 X D]
    for i in range(1, K + 1):
        Uo_neg = outputVectors[indices[i]]
        neg_x = -1 * np.dot(Uo_neg, Vc)
        grad[indices[i]] += - (sigmoid(neg_x) - 1) * Vc # [1 X D]

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens (V)
    outputVectors -- "output" word vectors (as rows) for all tokens (U)
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    predicted_index = tokens[currentWord]
    predicted = inputVectors[predicted_index]
    for j in range(min(2 * C, len(contextWords))):
        target_word_index = tokens[contextWords[j]]
        costF, gradPredF, gradF = word2vecCostAndGradient(predicted, target_word_index, outputVectors, dataset)
        cost +=  costF
        gradOut += gradF
        gradIn[predicted_index] += gradPredF


    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)


    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in xrange(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
