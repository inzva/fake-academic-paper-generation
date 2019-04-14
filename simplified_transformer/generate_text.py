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
parser.add_argument('--chars_to_generate', type=int, default=1000, help='')
parser.add_argument('--temperature', type=float, default=None, help='')
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

# Read, then decode for py2 compat.
text = open(opt.input_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))

# Take a look at the first 250 characters in text
print(text[:250])

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

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


model = build_model(vocab_size, opt.seq_length+1, 1, d_model=opt.d_model,d_inner_hid=opt.d_inner_hid,\
    n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, layers=opt.layers, dropout=opt.dropout, active_layers=opt.active_layers)

# Directory where the checkpoints will be saved
checkpoint_dir = './experiment/training_checkpoints_seq_len_{}'.format(opt.seq_length)

#RecursionError: maximum recursion depth exceeded while loading parameters
sys.setrecursionlimit(10000)
tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()


def generate_text(model, start_string, temperature):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = opt.chars_to_generate

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    # temperature = 0.5

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

if opt.temperature is None:
    temperatures = [0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.]
else:
    temperatures = [opt.temperature]

for temperature in temperatures:
    with open(os.path.join(checkpoint_dir, 'generated_text_temp_{}.txt'.format(temperature)), 'w+', encoding='utf-8') as f:
        print(generate_text(model, start_string=u"\\begin{document}", temperature=temperature), file=f)
