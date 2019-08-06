import os
from collections import Counter, OrderedDict

import torch


class Vocab(object):
    EOS = '<eos>'

    def __init__(self, special=(), min_freq=0, max_size=None, lower_case=True,
                 vocab_file=None,
                 add_eos=False, add_double_eos=False):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.vocab_file = vocab_file
        self.add_eos = add_eos
        self.add_double_eos = add_double_eos

    @classmethod
    def from_symbols(cls, symbols, **kwargs):
        vocab = cls(**kwargs)
        vocab.set_symbols(symbols)
        return vocab

    def tokenize(self, line):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        symbols = line.split()

        if self.add_double_eos: # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif self.add_eos:
            return symbols + [self.EOS]
        else:
            return symbols

    def count_file(self, path, verbose=False):
        if verbose: print('counting file {} ...'.format(path))
        assert os.path.exists(path)

        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line)
                self.counter.update(symbols)

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        symbols = []
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symbols.append(line.strip().split()[0])
        self.set_symbols(symbols)

    def set_symbols(self, symbols):
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        for sym in symbols:
            self.add_symbol(sym)
        self.unk_idx = self.sym2idx.get('<UNK>')
        self.unk_idx = 0

    def build_vocab(self):
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))

    def encode_file(self, path, ordered=False, verbose=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line)
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert self.EOS not in sym
            assert self.unk_idx is not None
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices
                             if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)
