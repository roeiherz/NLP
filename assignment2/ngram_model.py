#!/usr/local/bin/python
import numpy

from data_utils import utils as du
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                      index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """

    token_count = 0
    trigram_count_dict = {}
    bigram_count_dict = {}
    unigram_count_dict = {}
    for sentence in dataset:
        # Token counter
        token_count += sentence.shape[0]

        # Count the trigram model
        first_word = sentence[:-2]
        second_word = sentence[1:-1]
        third_word = sentence[2:]
        for i in range(len(first_word)):
            trigram = (first_word[i], second_word[i], third_word[i])
            if trigram in trigram_count_dict:
                trigram_count_dict[trigram] += 1
            else:
                trigram_count_dict[trigram] = 1

        # Count the bigram model
        first_word = sentence[:-1]
        second_word = sentence[1:]
        for i in range(len(first_word)):
            brigram = (first_word[i], second_word[i])
            if brigram in bigram_count_dict:
                bigram_count_dict[brigram] += 1
            else:
                bigram_count_dict[brigram] = 1

        # Count the unigram model
        for i in range(len(sentence)):
            unigram = (sentence[i])
            if unigram in unigram_count_dict:
                unigram_count_dict[unigram] += 1
            else:
                unigram_count_dict[unigram] = 1

    return trigram_count_dict, bigram_count_dict, unigram_count_dict, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """

    #
    perplexity_log = 0
    # Number of sentences
    M = 0
    for sentence in eval_dataset:

        # Arrange the arrays
        first_word = sentence[:-2]
        second_word = sentence[1:-1]
        third_word = sentence[2:]
        prob_sentence = 1.0
        M += len(sentence)
        for i in range(len(first_word)):
            # Get the models
            trigram = (first_word[i], second_word[i], third_word[i])
            bigram_u_v = (first_word[i], second_word[i])
            bigram_v_w = (second_word[i], third_word[i])
            unigram_v = (second_word[i])
            unigram_w = (third_word[i])

            if trigram_counts.has_key(trigram):
                prob_trigram = float(trigram_counts[trigram]) / bigram_counts[bigram_u_v]
            else:
                prob_trigram = 0

            if bigram_counts.has_key(bigram_v_w):
                prob_bigram = float(bigram_counts[bigram_v_w]) / unigram_counts[unigram_v]
            else:
                prob_bigram = 0

            if unigram_counts.has_key(unigram_w):
                prob_unigram = float(unigram_counts[unigram_w]) / train_token_count
            else:
                prob_unigram = 0

            q_ML = lambda1 * prob_trigram + lambda2 * prob_bigram + (1 - lambda1 -lambda2) * prob_unigram
            prob_sentence *= q_ML

        perplexity_log += np.log2(prob_sentence)

    perplexity = pow(2, -1 *  perplexity_log / M)

    return perplexity


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    # Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    ### END YOUR CODE

def lambda_grid_search():
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)

    perplexities1 = []
    lambdas1 = []
    lambda1_list = [t/1000.0 for t in range(350,500)]
    lambda2 = 0.4
    for lambda1 in lambda1_list:
        if lambda1 + lambda2 >= 1:
            continue
        perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)
        perplexities1.append(perplexity)
        lambdas1.append(lambda1)

    best_index = numpy.argmin(perplexities1)
    lambda1 = lambdas1[best_index]
    print("best lambda1 is {0}".format(lambda1))

    plot_graph(perplexities1, lambdas1, "grid_search_lambda1", "perplexity1", "lambda1", "perplexity")

    lambda2_list = [t/1000.0 for t in range(350,500)]
    perplexities2 = []
    lambdas2 = []
    for lambda2 in lambda2_list:
        if lambda1 + lambda2 >= 1:
            continue
        perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)
        perplexities2.append(perplexity)
        lambdas2.append(lambda2)

    best_index = numpy.argmin(perplexities2)
    lambda2 = lambdas2[best_index]
    print("best lambda2 is {0}".format(lambda2))

    print("best preplexity is {0}".format(perplexities2[best_index]))


    plt.figure()
    plot_graph(perplexities2, lambdas2, "grid_search_lambda2", "perplexity2", "lambda2", "perplexity")

def plot_graph(y_lst, x_lst, file_name='', label='', title='', ylabel='', xlabel=''):
    """
    This function is plotting the graph
    :param label: for plotting legend
    :param title: title for the plotting
    :param ylabel: ylabel for the plotting
    :param xlabel: xlabel for the plotting
    :param error_lst: the of errors
    :param x_lst: list of the number of samples
    :param file_name: file_name to be saved
    """
    plt.plot(x_lst, y_lst, label=label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig('{}.png'.format(file_name))


if __name__ == "__main__":
    test_ngram()
    lambda_grid_search()