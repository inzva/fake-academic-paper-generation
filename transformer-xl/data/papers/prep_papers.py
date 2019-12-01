#!/usr/bin/env python
# coding=utf-8

import os
import sys

if os.path.exists('train.txt'):
    print('Tokenized papers already exists - skipping processing')
    sys.exit()

data = open('preprocessed_data.txt', 'rb').read()

print('Length of papers: {}'.format(len(data)))

train_data = data[: -3793068]
valid_data = data[-3793068:]

for fn, part in [('train.txt', train_data), ('valid.txt', valid_data)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Tokenizing...')
    part_str = ' '.join([str(c) if c != ord('\n') else str(c)+'\n' for c in part])
    print('- Writing...')
    f = open(fn, 'w').write(part_str)
    f = open(fn + '.raw', 'wb').write(part)
