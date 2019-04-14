import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seq_length', type=int, default=100, help='Input sequence length given to the network')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Size of the training batches')
parser.add_argument('--d_model', type=int, default=256, help='')
parser.add_argument('--d_inner_hid', type=int, default=512, help='')
parser.add_argument('--n_head', type=int, default=4, help='')
parser.add_argument('--d_k', type=int, default=64, help='')
parser.add_argument('--d_v', type=int, default=64, help='')
parser.add_argument('--layers', type=int, default=6, help='Number of stacked multi-head-layers layers')
parser.add_argument('--dropout', type=float, default=0.1, help='')
parser.add_argument('--active_layers', type=int, default=999, help='')
parser.add_argument('--input_file', type=str, default='../dataset/preprocessed_data.txt', help='Input file path')
opt = parser.parse_args()

import random, os, sys
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import keras.backend as K

from transformer import Encoder, GetPosEncodingMatrix


print("Arguments", opt)

# Read, then decode for py2 compat.
text = open(opt.input_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

# Show how the first 13 characters from the text are mapped to integers
print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# The maximum length sentence we want for a single input in characters
seq_length = opt.seq_length
examples_per_epoch = len(text) // seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# Batch size
BATCH_SIZE = opt.batch_size
steps_per_epoch = examples_per_epoch // BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

def build_model(n_tokens, len_limit, batch_size, d_model=256, d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=6, dropout=0.1, active_layers=999):
    d_emb = d_model

    pos_emb = Embedding(len_limit, d_emb, trainable=False, \
                        weights=[GetPosEncodingMatrix(len_limit, d_emb)], \
                            batch_input_shape=[batch_size, None])

    word_emb = Embedding(n_tokens, d_emb, batch_input_shape=[batch_size, None])

    encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                        word_emb=word_emb, pos_emb=pos_emb)
    target_layer = TimeDistributed(Dense(n_tokens, use_bias=False))

    def get_pos_seq(x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    src_seq = Input(shape=(None,), dtype='int32')

    src_pos = Lambda(get_pos_seq)(src_seq)

    enc_output = encoder(src_seq, src_pos, active_layers=active_layers)
    final_output = target_layer(enc_output)

    model = Model(inputs=src_seq, outputs=final_output)
    return model


model = build_model(vocab_size, seq_length+1, BATCH_SIZE, d_model=opt.d_model,d_inner_hid=opt.d_inner_hid,\
    n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, layers=opt.layers, dropout=opt.dropout, active_layers=opt.active_layers)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()


def loss(labels, logits):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=loss)


# Directory where the checkpoints will be saved
checkpoint_dir = './experiment/training_checkpoints_seq_len_{}'.format(opt.seq_length)
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

os.makedirs(checkpoint_dir, exist_ok=True)
#RecursionError: maximum recursion depth exceeded while saving parameters
sys.setrecursionlimit(10000)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = opt.epochs

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
