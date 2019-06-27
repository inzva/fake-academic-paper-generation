import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default="experiment")
parser.add_argument('--model', type=str, default="transformer")
parser.add_argument('--hparams_set', type=str, default="transformer_small")
opt = parser.parse_args()

folder = os.path.join(opt.folder, opt.model, opt.hparams_set)
tmp_dir = os.path.join(folder, "tmp")
data_dir = os.path.join(folder, "data")
output_dir = os.path.join(folder, "output")

command= ("t2t-trainer " + \
"--generate_data " + \
"--tmp_dir={} ".format(tmp_dir) + \
"--data_dir={} ".format(data_dir) + \
"--output_dir={} ".format(output_dir) + \
"--problem=paper_generation_problem " + \
"--t2t_usr_dir=t2t_paper_generation_problem " + \
"--model={} ".format(opt.model) + \
"--hparams_set={} ".format(opt.hparams_set))

try:
    os.system(command)
except KeyboardInterrupt:
    print("--> Train Interrupted by Keyboard. Files saved at", folder)
except Exception:
    pass
else:
    print("--> Train Completed. Files saved at", folder)