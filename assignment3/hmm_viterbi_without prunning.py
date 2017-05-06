from data import *
import numpy


def get_e_probs(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns dict that maps between tuple (word, tag) and the Number of times that the state s
        is seen paired with observation x in the corpus
    """

    # Number of times that the state s is seen paired with observation x in the corpus
    e_word_tag_counts = {}

    for sentence in dataset:

        for word_to_tag in sentence:
            # Foreach (word, tag) tuple we are calculating number of incstances
            if word_to_tag in e_word_tag_counts:
                e_word_tag_counts[word_to_tag] += 1
            else:
                e_word_tag_counts[word_to_tag] = 1

    return e_word_tag_counts


def get_q_probs(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """

    total_tokens = 0
    trigram_count_dict = {}
    bigram_count_dict = {}
    unigram_count_dict = {}
    for sentence in dataset:
        # Tag Token counter
        total_tokens += len(sentence)
        # Make the sentence will be only tags
        sentence = [word[1] for word in sentence]

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

    # Get the unique taggs
    e_tag_counts = unigram_count_dict.keys()
    return total_tokens, trigram_count_dict, bigram_count_dict, unigram_count_dict, e_tag_counts


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Rerutns: the q-counts and e-counts of the sentences' tags
    """
    # Get the counts for calculating p
    e_word_tag_counts = get_e_probs(sents)
    # Get the counts for calculating q
    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_tag_counts = get_q_probs(sents)

    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def get_q(tag_v, tag_w, tag_u, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts):
    """
    This function calculates q
    :param q_uni_counts: dict of unigrams
    :param q_bi_counts: dict of bigrams
    :param q_tri_counts: dict of trigrams
    :param total_tokens: integer total tokens
    :param tag_v: v tag
    :param tag_w: w tag
    :param tag_u: u tag
    :return:
    """
    lambda1 = 0.3
    lambda2 = 0.3
    lambda3 = 1 - lambda1 - lambda2

    q_tri = 0
    q_bi = 0
    q_uni = 0

    if (tag_w, tag_u, tag_v) in q_tri_counts:
        q_tri = float(q_tri_counts[(tag_w, tag_u, tag_v)]) / q_bi_counts[(tag_w, tag_u)]
    if (tag_u, tag_v) in q_bi_counts:
        q_bi = float(q_bi_counts[(tag_u, tag_v)]) / q_uni_counts[tag_u]
    if tag_v in q_uni_counts:
        q_uni = float(q_uni_counts[tag_v]) / total_tokens



    # Calculates the total 3 n-grams
    total_q = lambda1 * q_tri + lambda2 * q_bi + lambda3 * q_uni
    return total_q


def get_e(word, tag, e_word_tag_counts, q_uni_counts):
    """
    This function calculates the probability
    :param word: word
    :param tag: tag of the word
    :param e_word_tag_counts: dict of tuples of (word, tag)
    :param q_uni_counts: dict of tags and their number instances
    :return: the e probability
    """
    word_tag_tupple = (word, tag)

    word_tag_count = 0
    if word_tag_tupple in e_word_tag_counts:
        word_tag_count = e_word_tag_counts[word_tag_tupple]

    nof_tag = q_uni_counts[tag]
    return float(word_tag_count) / nof_tag


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """

    # Sentence length
    sent_length = len(sent)

    # Number of tags
    nof_tags = len(e_tag_counts)

    # predicted_tags = [""] * (sent_length)
    predicted_tags = []
    # Create phi matrix
    phi = numpy.zeros(shape=(sent_length, nof_tags, nof_tags))
    # Back Pointer
    bp = numpy.zeros(shape=(sent_length, nof_tags, nof_tags), dtype="int")

    for k in range(sent_length):
        for v in range(nof_tags):
            # Tag V
            tag_v = e_tag_counts[v]
            for u in range(nof_tags):
                # Tag U
                tag_u = e_tag_counts[u]
                max_w_temp = 0
                max_w_tag = 0
                for w in range(nof_tags):
                    # Tag W
                    tag_w = e_tag_counts[w]

                    # For padding case
                    phi_val = 1
                    if k - 1 >= 0:
                        phi_val = phi[k - 1][w][u]

                    # Optional the W tag that we are looking for
                    w_temp_val = phi_val * get_q(tag_v, tag_w, tag_u, total_tokens, q_tri_counts, q_bi_counts,
                                                 q_uni_counts) * get_e(sent[k], tag_v, e_word_tag_counts,
                                                                       q_uni_counts)

                    # Update maximum
                    if max_w_temp < w_temp_val:
                        max_w_temp = w_temp_val
                        max_w_tag = w

                phi[k][u][v] = max_w_temp
                bp[k][u][v] = max_w_tag

    # Find the maximum tags with the bp matrix
    phi_max = numpy.argmax(phi[k])
    # Col index
    v_index = phi_max % nof_tags
    # Row index
    u_index = phi_max / nof_tags
    w_index = bp[k][u_index][v_index]
    predicted_tags.append(e_tag_counts[v_index])
    predicted_tags.append(e_tag_counts[u_index])
    predicted_tags.append(e_tag_counts[w_index])
    for k in range(sent_length - 2, 1, -1):
        v_index = u_index
        u_index = w_index
        w_index = bp[k][u_index][v_index]
        predicted_tags.append(e_tag_counts[w_index])

    # Reverse the tag list
    predicted_tags.reverse()
    return predicted_tags


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """

    correct_tag = 0
    total_tag = 0

    debug_inx = 0
    for sent in test_data:
        word_sentence = [word[0] for word in sent]
        tag_sentence = [word[1] for word in sent]
        predicated_sent = hmm_viterbi(word_sentence, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                      e_word_tag_counts,
                                      e_tag_counts)

        # Evaluate the number of correct tags
        for i in range(len(tag_sentence)):
            if tag_sentence[i] == predicated_sent[i]:
                correct_tag += 1
            total_tag += 1

        # Only for debugging
        debug_inx += 1

        #if debug_inx == 3:
        #    break

    acc_viterbi = float(correct_tag) / total_tag
    return acc_viterbi


if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                           e_tag_counts)
    print "dev: acc hmm viterbi: " + str(acc_viterbi)

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                               e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + str(acc_viterbi)
