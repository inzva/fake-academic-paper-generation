import pickle, os
import numpy as np
import tensorflow as tf
# Enable Eager execution - useful for seeing the generated data.
tf.enable_eager_execution()

from PaperGenerationProblem import *

DATA_DIR = "./t2t_problem_test/data"
TMP_DIR = "./t2t_problem_test/tmp"
OUTPUT_DIR = "./t2t_problem_test/output"

if os.path.exists(DATA_DIR):
    tf.gfile.DeleteRecursively(DATA_DIR)

# Create them.
tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(OUTPUT_DIR)
tf.gfile.MakeDirs(TMP_DIR)


paper_generation_problem = PaperGenerationProblem()
SEQ_LEN = paper_generation_problem.sequence_length
paper_generation_problem.generate_data(DATA_DIR, TMP_DIR)

with open(os.path.join(TMP_DIR, 'paper_dataset.txt'), 'rb') as f:
    paper_dataset = f.read().decode(encoding='utf-8')

tfe = tf.contrib.eager
Modes = tf.estimator.ModeKeys
# We can iterate over our examples by making an iterator and calling next on it.
eager_iterator = tfe.Iterator(paper_generation_problem.dataset(Modes.TRAIN, DATA_DIR))

example = eager_iterator.next()
target_tensor = example["targets"]
test_line = target_tensor.numpy()
encoder = paper_generation_problem._encoders['targets']
test_line_text = encoder.decode(test_line)

print("Number of Samples in the Dataset: {}".format(paper_generation_problem.nb_samples))
try:
    assert paper_dataset.find(test_line_text) != -1
    print("--> TEST RESULT: SUCCEED")
except:
    print("--> TEST RESULT: FAILED")