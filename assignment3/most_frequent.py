from data import *
import operator


def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """

    # For each word we will have a dict with his tags freq
    word_tag_count = {}
    for line in train_data:
        for word_tuple in line:
            # Get the word and the tag
            word = word_tuple[0]
            tag = word_tuple[1]

            # Check if the word is already exist in the dict
            if word in word_tag_count:
                tag_count = word_tag_count[word]
            else:
                tag_count = {}
                word_tag_count[word] = tag_count

            # Update the counts of the tag per
            if tag in tag_count:
                tag_count[tag] += 1
            else:
                tag_count[tag] = 1

    # Find the word and her max tag
    max_word_tag = {}
    for word in word_tag_count:
        max_word_tag[word] = max(word_tag_count[word].iteritems(), key=operator.itemgetter(1))[0]

    return max_word_tag


def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """

    total_word_num = 0
    number_correct_tags = 0
    # For each sentence in the test data-set we are going to check each tag per word and see if its tag is also
    # the max freq tag that we were calculated in the train data-set
    for sentence in test_set:
        for word_tuple in sentence:
            word = word_tuple[0]
            tag = word_tuple[1]
            # If the tag in the test data-set is also the the max tag freq
            if word in pred_tags and pred_tags[word] == tag:
                number_correct_tags += 1
            total_word_num += 1

    accuracy = number_correct_tags / float(total_word_num)
    return accuracy


if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + most_frequent_eval(dev_sents, model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)
