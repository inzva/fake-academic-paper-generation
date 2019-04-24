import os
import argparse 
import numpy as np
import six

from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--sampling_temperature', type=float, default=0.6)
parser.add_argument('--folder', type=str, default="experiment")
parser.add_argument('--model', type=str, default="transformer", choices=["transformer"])
parser.add_argument('--hparams_set', type=str, default="transformer_small", choices=["transformer_small"])
opt = parser.parse_args()


folder = os.path.join(opt.folder, opt.model, opt.hparams_set)
tmp_dir = os.path.join(folder, "tmp")
data_dir = os.path.join(folder, "data")
output_dir = os.path.join(folder, "output")

generated_paper_path = os.path.join(output_dir, "generated_paper_{}.txt".format(opt.seq_len))
os.makedirs(os.path.dirname(generated_paper_path), exist_ok=True)

# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
                    "must be in the format input \t target.")
flags.DEFINE_bool("decode_in_memory", False, "Decode in memory.")

FLAGS.tmp_dir = tmp_dir
FLAGS.data_dir = data_dir
FLAGS.output_dir = output_dir
FLAGS.problem = "paper_generation_problem"
FLAGS.t2t_usr_dir = "t2t_paper_generation_problem"
FLAGS.model = opt.model
FLAGS.hparams_set = opt.hparams_set
FLAGS.decode_hparams="beam_size=1,alpha=0.6"

def create_hparams():
   return trainer_lib.create_hparams(
         FLAGS.hparams_set,
         FLAGS.hparams,
         data_dir=os.path.expanduser(FLAGS.data_dir),
         problem_name=FLAGS.problem)

def create_decode_hparams():
  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.shards = FLAGS.decode_shards
  decode_hp.shard_id = FLAGS.worker_id
  decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
  decode_hp.decode_in_memory = decode_in_memory
  decode_hp.decode_to_file = FLAGS.decode_to_file
  decode_hp.decode_reference = FLAGS.decode_reference
  return decode_hp

trainer_lib.set_random_seed(17)
usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

hp = create_hparams()
decode_hp = create_decode_hparams()
hp.sampling_method="random"
hp.sampling_temp=opt.sampling_temperature

estimator = trainer_lib.create_estimator(
   FLAGS.model,
   hp,
   t2t_trainer.create_run_config(hp),
   decode_hparams=decode_hp,
   use_tpu=FLAGS.use_tpu)

problem = hp.problem

def _interactive_input_tensor_to_features_dict(feature_map, hparams):
   """Convert the interactive input format (see above) to a dictionary.
   Args:
      feature_map: dict with inputs.
      hparams: model hyperparameters
   Returns:
      a features dictionary, as expected by the decoder.
   """
   inputs = tf.convert_to_tensor(feature_map["inputs"])

   x = inputs
   # Remove the batch dimension.
   num_samples = x[0]
   length = x[2]
   x = tf.slice(x, [3], tf.to_int32([length]))
   x = tf.reshape(x, [1, -1, 1, 1])
   # Transform into a batch of size num_samples to get that many random
   # decodes.
   x = tf.tile(x, tf.to_int32([num_samples, 1, 1, 1]))

   p_hparams = hparams.problem_hparams
   input_space_id = tf.constant(p_hparams.input_space_id)
   target_space_id = tf.constant(p_hparams.target_space_id)

   features = {}
   features["input_space_id"] = input_space_id
   features["target_space_id"] = target_space_id
   features["decode_length"] = inputs[1]
   features["inputs"] = x
   return features

def _interactive_input_fn(hparams, decode_length=1024, input_string="\documentclass"):
   num_samples = 1
   input_type = "text"
   p_hparams = hparams.problem_hparams
   has_input = "inputs" in p_hparams.modality
   vocabulary = p_hparams.vocabulary["inputs" if has_input else "targets"]
   # This should be longer than the longest input.
   const_array_size = 10000
  
   input_ids = vocabulary.encode(input_string)
   if has_input:
      input_ids.append(text_encoder.EOS_ID)
   x = [num_samples, decode_length, len(input_ids)] + input_ids
   assert len(x) < const_array_size
   x += [0] * (const_array_size - len(x))
   features = {
      "inputs": np.array(x).astype(np.int32),
   }
   for k, v in six.iteritems(problem_lib.problem_hparams_to_features(p_hparams)):
      features[k] = np.array(v).astype(np.int32)
   yield features

def make_input_fn_from_generator(gen):
  """Use py_func to yield elements from the given generator."""
  first_ex = six.next(gen)
  flattened = tf.contrib.framework.nest.flatten(first_ex)
  types = [t.dtype for t in flattened]
  shapes = [[None] * len(t.shape) for t in flattened]
  first_ex_list = [first_ex]

  def py_func():
    if first_ex_list:
      example = first_ex_list.pop()
    else:
      example = six.next(gen)
    return tf.contrib.framework.nest.flatten(example)

  def input_fn():
    flat_example = tf.py_func(py_func, [], types)
    _ = [t.set_shape(shape) for t, shape in zip(flat_example, shapes)]
    example = tf.contrib.framework.nest.pack_sequence_as(first_ex, flat_example)
    return example

  return input_fn

vocabulary = hp.problem_hparams.vocabulary["targets"]

output_text = "\documentclass"
while len(output_text) < opt.seq_len:
   def input_fn():
      gen_fn = make_input_fn_from_generator(
         _interactive_input_fn(hp, decode_length=128, input_string=output_text))
      example = gen_fn()
      example = _interactive_input_tensor_to_features_dict(example, hp)
      return example

   prediction = list(estimator.predict(input_fn))[0]
   outputs = prediction["outputs"]
   if len(outputs) == 0:
      print("-> Failed to Generate Full Length Paper")
      break
   new_text = vocabulary.decode(outputs)
   output_text = output_text + new_text

with open(generated_paper_path, "wt") as f:
   f.write(output_text)

print("-> Paper Generated at ", generated_paper_path)