from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np
import os
import argparse

tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--seq_length', type=int, default=100, help='Embedding dimension')
parser.add_argument('--chars_to_generate', type=int, default=10000, help='Number of characters to be generated')
parser.add_argument('--recurrent_layers', type=int, default=1, help='Number of stacked recurrent layers')
parser.add_argument('--recurrent_units', type=int, default=1024, help='Number of recurrent units in each layer')
parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
opt = parser.parse_args()

filePath = 'dataset/preprocessed_data.txt'

# Read, then decode for py2 compat.
text = open(filePath, 'rb').read().decode(encoding='utf-8')
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

# The embedding dimension
embedding_dim = opt.embedding_dim

# Number of RNN units
rnn_units = opt.recurrent_units

if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNLSTM
else:
    import functools

    rnn = functools.partial(
        tf.keras.layers.LSTM, recurrent_activation='sigmoid')


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]))
    for i in range(opt.recurrent_layers):
        model.add(rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True))
    model.add(tf.keras.layers.Dense(vocab_size))

    return model


checkpoint_dir = '/mnt/apg-checkpoints/training_checkpoints_LSTM_HL_{}_HU_{}_seq_len_{}'.format(opt.recurrent_layers,
                                                                                                opt.recurrent_units,
                                                                                                opt.seq_length)

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

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


temperatures = [0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.]

for temperature in temperatures:
    with open(os.path.join(checkpoint_dir, 'generated_text_temp_{}.txt'.format(temperature)), 'w+', encoding='utf-8') as f:
        print(generate_text(model, start_string=u"\\begin{document}", temperature=temperature), file=f)
