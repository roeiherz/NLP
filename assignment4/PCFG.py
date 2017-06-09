from collections import defaultdict
import random


class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert (isinstance(lhs, str))
        assert (isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w, l, r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l, r, w)
        return grammar

    @classmethod
    def from_file_assert_cnf(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w, l, r = line.split(None, 2)
                r = r.split()
                w = float(w)
                if len(r) > 2:
                    raise Exception("Grammar is not CNF, right-hand-side is: " + str(r))
                if len(r) <= 0:
                    raise Exception("Grammar is not CNF, right-hand-side is empty: " + str(r))
                grammar.add_rule(l, r, w)
        for lhs, rhs_and_weight in grammar._rules.iteritems():
            rhs = rhs_and_weight[0]
            if len(rhs) == 1 and not grammar.is_terminal(rhs[0]):
                raise Exception("Grammar has unary rule: " + str(rhs))
        return grammar

    def is_terminal(self, symbol):
        return symbol not in self._rules

    def is_preterminal(self, rhs):
        return len(rhs) == 1 and self.is_terminal(rhs[0])

    def gen(self, symbol):
        if self.is_terminal(symbol):
            return symbol
        else:
            expansion = self.random_expansion(symbol)
            return " ".join(self.gen(s) for s in expansion)

    def gentree(self, symbol):
        """
            Generates a derivation tree from a given symbol
        """

        # Check that symbol is not terminal
        if self.is_terminal(symbol):
            return symbol

        # Get the expansion
        expansion = self.random_expansion(symbol)
        # Make recursively function
        seq = (symbol, str([self.gentree(s) for s in expansion]).replace('\\', '').replace("""'""", '').replace('[', '')
               .replace(']', '').replace(',', ''))
        space_between_seq = " ".join(seq)
        return '({})'.format(space_between_seq)

    def random_sent(self):
        return self.gen("ROOT")

    def random_tree(self):
        return self.gentree("ROOT")

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r, w in self._rules[symbol]:
            p = p - w
            if p < 0:
                return r
        return r
