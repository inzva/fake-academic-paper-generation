import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--folder', type=str, default="experiment")
parser.add_argument('--model', type=str, default="transformer", choices=["transformer"])
parser.add_argument('--hparams_set', type=str, default="transformer_small", choices=["transformer_small"])
opt = parser.parse_args()



folder = os.path.join(opt.folder, opt.model, opt.hparams_set)
tmp_dir = os.path.join(folder, "tmp")
data_dir = os.path.join(folder, "data")
output_dir = os.path.join(folder, "output")

tmp_file_path = os.path.join(tmp_dir, "tmp.txt")
os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)

generated_paper_path = os.path.join(output_dir, "generated_paper_{}.txt".format(opt.seq_len))
os.makedirs(os.path.dirname(generated_paper_path), exist_ok=True)

command= ("t2t-decoder " + \
"--tmp_dir={} ".format(tmp_dir) + \
"--data_dir={} ".format(data_dir) + \
"--output_dir={} ".format(output_dir) + \
"--problem=paper_generation_problem " + \
"--t2t_usr_dir=t2t_paper_generation_problem " + \
"--model={} ".format(opt.model) + \
"--hparams_set={} ".format(opt.hparams_set) + \
"--decode_from_file={} ".format(generated_paper_path) + \
"--decode_to_file={} ".format(tmp_file_path))


def calc_len():
   with open(generated_paper_path, "rt") as f: 
      return len(f.read())

def concat_files():
   with open(generated_paper_path, "a+t") as f: 
      with open(tmp_file_path, "rt") as f2: 
         f.write(f2.read())

with open(generated_paper_path, "wt") as f: 
   f.write("\documentclass")

while calc_len() < opt.seq_len:
   os.system(command)
   concat_files()

print("-> Paper Generated at ", generated_paper_path)