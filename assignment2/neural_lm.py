#!/usr/local/bin/python
import cPickle
import pandas as pd
import time
import os.path

from data_utils import utils as du
from numpy import *
from neural import *
from sgd import *

VOCAB_EMBEDDING_PATH = "data/lm/vocab.embeddings.glove.txt"
BATCH_SIZE = 50
NUM_OF_SGD_ITERATIONS = 40000
LEARNING_RATE = 0.3


def load_vocab_embeddings(path=VOCAB_EMBEDDING_PATH):
    result = []
    with open(path) as f:
        index = 0
        for line in f:
            line = line.strip()
            row = line.split()
            data = [float(x) for x in row[1:]]
            assert len(data) == 50
            result.append(data)
            index += 1
    return result


def load_data_as_sentences(path, word_to_num):
    """
    Conv:erts the training data to an array of integer arrays.
      args: 
        path: string pointing to the training data
        word_to_num: A dictionary from string words to integers
      returns:
        An array of integer arrays. Each array is a sentence and each 
        integer is a word.
    """
    docs_data = du.load_dataset(path)
    S_data = du.docs_to_indices(docs_data, word_to_num)
    return docs_data, S_data


def convert_to_lm_dataset(S):
    """
    Takes a dataset that is a list of sentences as an array of integer arrays.
    Returns the dataset a bigram prediction problem. For any word, predict the
    next work. 
    IMPORTANT: we have two padding tokens at the beginning but since we are 
    training a bigram model, only one will be used.
    """
    in_word_index, out_word_index = [], []
    for sentence in S:
        for i in xrange(len(sentence)):
            if i < 2:
                continue
            in_word_index.append(sentence[i - 1])
            out_word_index.append(sentence[i])
    return in_word_index, out_word_index


def shuffle_training_data(in_word_index, out_word_index):
    combined = zip(in_word_index, out_word_index)
    random.shuffle(combined)
    return zip(*combined)


def int_to_one_hot(number, dim):
    res = np.zeros(dim)
    res[number] = 1.0
    return res


def lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, params):

    # Construct the data batch
    in_word_index_arr = np.array(in_word_index)
    out_word_index_arr = np.array(out_word_index)
    num_to_word_embedding_arr = np.array(num_to_word_embedding)

    rand_indice = np.random.randint(0, len(in_word_index), BATCH_SIZE)
    words = in_word_index_arr[rand_indice]
    data = num_to_word_embedding_arr[words]     # [BATCH_SIZE=50, EMBEDDING_SIZE=50]
    l = out_word_index_arr[rand_indice]      # [BATCH_SIZE=50]
    # Labels is a one hot matrix [BATCH_SIZE=50, 2000]
    labels = np.zeros((BATCH_SIZE, dimensions[2]))
    labels[np.arange(BATCH_SIZE), l] = 1

    # Construct the data batch and run you backpropogation implementation
    cost, grad = forward_backward_prop(data, labels, params, dimensions)

    cost /= BATCH_SIZE
    grad /= BATCH_SIZE
    return cost, grad


def eval_neural_lm(eval_data_path):
    """
    Evaluate perplexity (use dev set when tuning and test at the end)
    """
    _, S_dev = load_data_as_sentences(eval_data_path, word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_dev)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    perplexity = 0
    ### YOUR CODE HERE
    num_to_word_embedding_arr = np.array(num_to_word_embedding)


    data = num_to_word_embedding_arr[in_word_index]
    # Labels is a one hot matrix
    labels = np.zeros((num_of_examples, dimensions[2]))
    labels[np.arange(num_of_examples), out_word_index] = 1

    num_of_batches = num_of_examples / BATCH_SIZE
    num_of_examples = BATCH_SIZE * num_of_batches

    sum_log_probs = 0
    for i in range(num_of_batches):
        start = i * BATCH_SIZE
        end = (i + 1) * BATCH_SIZE
        probs = forward(data[start:end], labels[start:end], params, dimensions)

        log_probs = np.log2(probs)
        sum_log_probs += np.sum(log_probs)

    perplexity = pow(2, -1 *  sum_log_probs / num_of_examples)


    ### END YOUR CODE

    return perplexity


if __name__ == "__main__":
    # Load the vocabulary
    vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                          index_col=0, names=['count', 'freq'], )

    vocabsize = 2000
    num_to_word = dict(enumerate(vocab.index[:vocabsize]))
    num_to_word_embedding = load_vocab_embeddings()
    word_to_num = du.invert_dict(num_to_word)

    # Load the training data
    _, S_train = load_data_as_sentences('data/lm/ptb-train.txt', word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_train)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    random.seed(31415)
    np.random.seed(9265)
    in_word_index, out_word_index = shuffle_training_data(in_word_index, out_word_index)
    startTime = time.time()

    # Training should happen here

    # Initialize parameters randomly
    # Construct the params
    input_dim = 50
    hidden_dim = 50
    output_dim = vocabsize
    dimensions = [input_dim, hidden_dim, output_dim]
    params = np.random.randn((input_dim + 1) * hidden_dim + (
        hidden_dim + 1) * output_dim, )

    print "#params: " + str(len(params))
    print "#train examples: " + str(num_of_examples)

    # run SGD
    params = sgd(
        lambda vec: lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, vec),
        params, LEARNING_RATE, NUM_OF_SGD_ITERATIONS, None, True, 1000)

    print "training took %d seconds" % (time.time() - startTime)



    # Evaluate perplexity with dev-data
    perplexity = eval_neural_lm('data/lm/ptb-dev.txt')
    print "dev perplexity : " + str(perplexity)

    # Evaluate perplexity with test-data (only at test time!)
    if os.path.exists('data/lm/ptb-test.txt'):
        perplexity = eval_neural_lm('data/lm/ptb-test.txt')
        print "test perplexity : " + str(perplexity)
    else:
        print "test perplexity will be evaluated only at test time!"
