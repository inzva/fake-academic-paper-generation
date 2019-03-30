import re, os
from tqdm import tqdm
import codecs
import sys

def preprocess_dataset(use_vocab=False, use_lower=False, use_blacklist=False,
                       blacklist_file='blacklist.txt', blacklist_threshold=100):

    if not os.path.exists('./dataset'):
        os.mkdir('./dataset')

    vocab = {}

    def add_to_vocab(string):
        for char in string:
            if char in vocab: vocab[char] += 1
            else: vocab[char] = 1

    def preprocess_paper(raw_text):
        output = ''
        raw_text = raw_text.split('\n')

        # Find lines start with % which means comment in laTex, replace them lines with newlines
        raw_text = ['\n' if i is '' else i for i in raw_text if len(re.findall('^\s{0,}%', i)) == 0]

        for line in raw_text:
            if use_lower: line = line.lower()
            if use_vocab: add_to_vocab(line)
            output += (line + '\n')

        if use_vocab: return output, vocab
        else: return output

    # Creating blacklist that contains items lower than threshold and write it in RegEx format
    # There can be needed manual replacements for non-english characters
    def create_blacklist(vocab):
        blacklist = [i for i in vocab if vocab[i] < blacklist_threshold]
        with open(out_file, 'wb') as file:
            file.write('|'.join(blacklist_file).encode('utf-8'))


    for file_name in tqdm(os.listdir('dataset_generation/papers'),ascii=True):
        with open('dataset_generation/papers/%s' % file_name, 'rb') as file:
            paper_text = file.read().decode('utf-8', 'ignore')

            if use_vocab: (output, vocab) = preprocess_paper(paper_text)
            else: output = preprocess_paper(paper_text)

        with open('dataset/%s' % file_name, 'wb') as file:
            file.write(output.encode('utf-8'))

    print('>>Char based dataset is created...')
    if use_vocab: return vocab


def read_blacklist(file_name):
    with open(file_name, 'rb') as file:
        blacklist = file.read().decode('utf-8').split('|')
    return blacklist
