import os
import re

MIN_FREQ = 3


def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res


def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1], tokens[3]))
    return sents


def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab


def replace_word(word):
    """
        Replaces rare words with ctegories (numbers, dates, etc...) based on paper 
        http://people.csail.mit.edu/mcollins/6864/slides/bikel.pdf (page 7)
    """
    try:

        # twoDigitNum class: "90"
        if word.isdigit() and len(word) == 2:
            # print "twoDigitNum: " + word
            return "twoDigitNum"

        # fourDigitNum Class: "1987"
        if word.isdigit() and len(word) == 4:
            # print "fourDigitNum: " + word
            return "fourDigitNum"

        # containsDigitAndAlpha class: "A4618-02"
        if re.match("^A[0-9]+-[0-9]+$", word):
            # print "containsDigitAndAlpha: " + word
            return "containsDigitAndAlpha"

        # containsDigitAndDash class: "18-02"
        if re.match("^[0-9]+-[0-9]+$", word):
            # print "containsDigitAndDash: " + word
            return "containsDigitAndDash"

        # containsDigitAndSlash class: "18-02"
        if re.match("^[0-9]+/[0-9]+/[0-9]+$", word):
            # print "containsDigitAndSlash: " + word
            return "containsDigitAndSlash"

        # containsDigitAndComma class: "18,1987.0"
        if re.match("^[0-9,.]+$", word):
            # print "containsDigitAndComma: " + word
            return "containsDigitAndComma"

        # containsDigitAndPeriod class: "18.0"
        if re.match("^[0-9.]+$", word):
            # print "containsDigitAndPeriod: " + word
            return "containsDigitAndPeriod"

        # otherNum class: "18555"
        if re.match("^[0-9]+$", word):
            # print "otherNum: " + word
            return "otherNum"

        # allCaps class: "LSTM"
        if re.match("^[A-Z]+$", word):
            # print "allCaps: " + word
            return "allCaps"

        # first word in a sentence class
        # if not re.match("^[A-Z][.]$", word):
        #     return "firstWord"

        # initCap class: "Herzig"
        if re.match("^[A-Z][a-z]+$", word):
            # print "initCap: " + word
            return "initCap"

        # lowerCase class: "herzig"
        if re.match("^[a-z]+$", word):
            # print "lowerCase: " + word
            return "lowerCase"

        # print "##########other: " + word
        return "other"
    except Exception as e:
        print "Error for replacing word"
        print(str(e))


def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print "replaced: " + str(float(replaced) / total)
    return res
