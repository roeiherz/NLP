from PCFG import PCFG

if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file(sys.argv[1])
    num_of_sents = 1
    print_tree = False
    if len(sys.argv) > 2:
        assert sys.argv[2] == "-n"
        num_of_sents = int(sys.argv[3])
    if len(sys.argv) > 4:
        assert sys.argv[4] == "-t"
        for i in xrange(num_of_sents):
            print pcfg.random_tree()
    else:
        for i in xrange(num_of_sents):
            print pcfg.random_sent()
