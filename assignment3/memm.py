from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import numpy as np
import time


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    features['word'] = curr_word
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['tag_bigram'] = prevprev_tag + "-" + prev_tag
    features['prev_wordtag_pairs'] = prev_word + "/" + prev_tag
    features['prevprev_wordtag_pairs'] = prevprev_word + "/" + prevprev_tag
    features['next_word'] = next_word
#    features['']

    ### YOUR CODE HERE
#    raise NotImplementedError
    ### END YOUR CODE
    return features


def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1],
                                 prevprev_token[1])

def viterbi_extract_features(sentence, k, u, w):
    curr_word = sentence[k][0]
    prev_word = sentence[k - 1][0] if k > 0 else '<s>'
    prevprev_word = sentence[k - 2][0] if k > 1 else '<s>'
    next_word = sentence[k + 1][0] if k < (len(sentence) - 1) else '</s>'
    return extract_features_base(curr_word, next_word, prev_word, prevprev_word, u, w)


def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Rerutns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)


def create_examples(sents):
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tagset[sent[i][1]])
    return examples, labels
    print "done"


def memm_greeedy(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    sent_examples, sent_labels = create_examples(sent)
    sent_examples_vectorized = vec.transform(sent_examples, sent_labels)

    predicted_tags = logreg.predict(sent_examples_vectorized)

    # calc accuracy
    sent_labels_arr = np.array(sent_labels)
    correct_prediction = np.sum(sent_labels_arr == predicted_tags)
    accuracy = correct_prediction / float(len(sent_labels))

    return accuracy


def memm_viterbi(sents, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    # Number of tags
    nof_tags = len(tagset)

    # tag of all sentences

    predicted_tags = []

    for sentence in sents:
        # nof tokens
        sent_length = len(sentence)

        # prediction per sentence
        sentence_prediction = []

        # Create phi matrix
        phi = np.zeros(shape=(sent_length, nof_tags, nof_tags))
        # Back Pointer
        bp = np.zeros(shape=(sent_length, nof_tags, nof_tags), dtype="int")

        for k in range(sent_length):
            for tag_u in tagset:
                # U index
                u = tagset[tag_u]

                max_w_temp = np.zeros(len(tagset))
                max_w_tag = np.zeros(len(tagset))
                for tag_w in tagset:
                    # W index
                    w = tagset[tag_w]

                    # For padding case
                    phi_val = 1
                    if k - 1 >= 0:
                        phi_val = phi[k - 1][w][u]

                    # Optional the W tag that we are looking for
                    features = viterbi_extract_features(sentence, k, tag_u, tag_w)
                    vectorized = vec.transform(features)

                    prob = logreg.predict_proba(vectorized)

                    w_temp_val = phi_val * prob

                    # Update maximum
                    for tag_v in tagset:
                        # index V
                        v = tagset[tag_v]

                        if max_w_temp[v] < w_temp_val[0][v]:
                            max_w_temp[v] = w_temp_val[0][v]
                            max_w_tag[v] = w

                # Update maximum
                phi[k][u] = max_w_temp
                bp[k][u] = max_w_tag

        inv_tagset = {v: k for k, v in tagset.iteritems()}

        # Find the maximum tags with the bp matrix
        phi_max = np.argmax(phi[k])
        # Col index
        v_index = phi_max % nof_tags
        # Row index
        u_index = phi_max / nof_tags
        w_index = bp[k][u_index][v_index]
        sentence_prediction.append(v_index)
        sentence_prediction.append(u_index)
        sentence_prediction.append(w_index)
        for k in range(sent_length - 2, 1, -1):
            v_index = u_index
            u_index = w_index
            w_index = bp[k][u_index][v_index]
            sentence_prediction.append(w_index)

        # Reverse the tag list
        sentence_prediction.reverse()

        # append to global prediction
        predicted_tags += sentence_prediction

    # calc accuracy
    sent_examples, sent_labels = create_examples(sents)
    sent_labels_arr = np.array(sent_labels)
    correct_prediction = np.sum(sent_labels_arr == predicted_tags)
    accuracy = correct_prediction / float(len(sent_labels))

    return accuracy



def memm_eval(test_data, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """

    # Greedy
    acc_greedy = memm_greeedy(test_data, logreg, vec)

    # Viterbi
    acc_viterbi = memm_viterbi(test_data, logreg, vec)

    return acc_viterbi, acc_greedy


if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    # The log-linear model training.
    # NOTE: this part of the code is just a suggestion! You can change it as you wish!
    curr_tag_index = 0
    tagset = {}
    for train_sent in train_sents:
        for token in train_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1
    for dev_sent in dev_sents:
        for token in dev_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1

    index_to_tag_dict = invert_dict(tagset)
    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    # ###########FIXME !!!
    #train_labels[1] = 38

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print "Fitting..."
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print "done, " + str(end - start) + " sec"
    # End of log linear model training

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec)
    print "dev: acc memm greedy: " + str(acc_greedy)
    print "dev: acc memm viterbi: " + str(acc_viterbi)
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + str(acc_greedy)
        print "test: acc memmm viterbi: " + str(acc_viterbi)
