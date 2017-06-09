from PCFG import PCFG
import math


def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents


def get_probabilities(N, rules, sums):
    """
    This function returns q - the probabilities of each Non-terminal
    :param N: non-terminals list
    :param rules: dict of list of rules
    :param sums: dict of sum of the Non-terminals
    :return: dict of probabilities of each Non-terminal
    """
    q = {}
    for lhs in N:
        for rhs in rules[lhs]:
            key = (lhs, tuple(rhs[0]))
            value = rhs[1] / float(sums[lhs])
            q[key] = value

    return q


def parse(X, bp):
    """
    This function iteratively
    :param X: Tuple of the (i, j, X) which i and j indices in the words in the sentence and X is a non-terminal
    :param bp: a dict which record the rule X->YZ and the split s that leading to the highest scoring parse trees
    :return:
    """

    # Get the words
    i, j, r = X[0], X[1], X[2]
    if i == j:
        return "(%s %s)" % (r, sent[i])
    else:
        Y, Z = bp[(i, j, r)]
        return "(%s %s %s)" % (r, parse(Y, bp), parse(Z, bp))


def initialization(n, N, q, sent):
    """
    This function initializes the pi and bp matrices
    :param n: the length of the list of parser sentence
    :param N: a list of non-terminals
    :param q: dict of probabilities of each Non-terminal
    :param sent: dict of probabilities of each Non-terminal
    :return: pi and bp dicts
    """
    # Initialize the bp and pi dict
    tab = [(i, j, X) for i in range(0, n) for j in range(0, n) for X in N]
    pi = {key: val for (key, val) in [(entry, 0.) for entry in tab]}
    bp = {key: val for (key, val) in [(entry, "") for entry in tab]}

    # Update pi and bp according to q
    for i in xrange(0, n):
        # Get the i word
        word_i = sent[i]
        for non_terminal in N:
            R = (non_terminal, (word_i,))

            # Update pi
            if not (R in q):
                pi[(i, i, non_terminal)] = 0.
            else:
                pi[(i, i, non_terminal)] = q[R]

            # Update bp
            if not (R in q):
                bp[(i, i, non_terminal)] = ""
            else:
                bp[(i, i, non_terminal)] = [(i, i, word_i)]

    return pi, bp


def cky(pcfg, sent):
    """
    This function get the PCFG rules (grammar) and the Sentence, and return the derivation of
    the sentence with the highest probability
    :param pcfg: grammar
    :param sent: sentence
    :return:
    """

    # print Sentence
    print ("The Sentence is: '{}'".format(sent))

    sent_lst = sent.split(' ')
    # length of the list of parser sentence
    n = len(sent_lst)
    # Get the rules with their probabilities and the sum of the
    rules, sums = pcfg._rules, pcfg._sums
    # Get the non-terminals
    N = rules.keys()
    # Get the probabilities
    q = get_probabilities(N, rules, sums)

    # Initialize the bp and pi dict
    # pi is a dict which fills the scores for each rule X->YZ and the split s
    # bp is a dict which record the rule X->YZ and the split s that leading to the highest scoring parse trees
    pi, bp = initialization(n, N, q, sent_lst)

    # CKY dynamic algorithm - fill pi bottom-up
    for l in xrange(1, n):
        for i in xrange(0, n-l):
            j = i+l
            for X in N:
                max_score_temp = 0.
                for s in range(i, j):
                    for Y in N:
                        for Z in N:

                            R = (X, (Y, Z))
                            # If the rule is not in q continue
                            if not (R in q):
                                continue

                            # For each s in {i...j-1} s.t X -> YZ in R
                            score = q[R] * pi[(i, s, Y)] * pi[(s+1, j, Z)]
                            # Find the maximum
                            if score >= max_score_temp:
                                bp[(i, j, X)] = [(i, s, Y), (s+1, j, Z)]
                                pi[(i, j, X)] = score
                                max_score_temp = score

    # Inference
    # Initialize parameters
    i = 0
    j = n - 1
    X = 'ROOT'

    # Do the parsing only for a non negative values
    if pi[(i, j, X)] > 0:
        return parse((i, j, X), bp)

    return "FAILED TO PARSE!"


if __name__ == '__main__':
    import sys

    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent)
