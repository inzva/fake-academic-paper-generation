import pickle, os
import numpy as np
import tensorflow as tf
# Enable Eager execution - useful for seeing the generated data.
tf.enable_eager_execution()

from PaperGenerationProblem import *

DATA_DIR = "./t2t_problem_test/data"
TMP_DIR = "./t2t_problem_test/tmp"
OUTPUT_DIR = "./t2t_problem_test/output"

# Create them.
tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(OUTPUT_DIR)
tf.gfile.MakeDirs(TMP_DIR)


paper_generation_problem = PaperGenerationProblem()
SEQ_LEN = paper_generation_problem.sequence_length
paper_generation_problem.generate_data(DATA_DIR, TMP_DIR)

with open(os.path.join(TMP_DIR, 'paper_dataset.txt'), 'rb') as f:
    paper_dataset = f.read().decode(encoding='utf-8')

data_seq_len = SEQ_LEN - 1
nb_samples = int(np.ceil(len(paper_dataset)/data_seq_len))
number_of_chars_in_last_example = len(paper_dataset) - (nb_samples-1)*data_seq_len
last_line_text_original = paper_dataset[-number_of_chars_in_last_example:]

tfe = tf.contrib.eager
Modes = tf.estimator.ModeKeys
# We can iterate over our examples by making an iterator and calling next on it.
eager_iterator = tfe.Iterator(paper_generation_problem.dataset(Modes.TRAIN, DATA_DIR))

i = 0
last_line_found = False
while(True):
    try:
        example = eager_iterator.next()
    except Exception as e:
        break

    target_tensor = example["targets"]

    if not last_line_found and int(tf.math.count_nonzero(target_tensor)) < SEQ_LEN:
        last_line_found = True
        last_line = target_tensor.numpy()[:-1]
        encoder = paper_generation_problem._encoders['targets']
        last_line_text = encoder.decode(last_line)
        try:
            assert last_line_text == last_line_text_original
            print("--> TEST RESULT: SUCCEED")
        except:
            print("--> TEST RESULT: FAILED")
    i += 1

print("Number of Examples in the Dataset: ", i)