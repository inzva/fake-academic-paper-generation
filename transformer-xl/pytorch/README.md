## Introduction

- `*base.sh` are for the base models which can be run on a few GPUs.


## Prerequisite

- Pytorch 0.4: `conda install pytorch torchvision -c pytorch`


## Data Prepration

`bash getdata.sh`

## Training and Evaluation

#### Training with the academic papers dataset

- Make sure the machine have **4 GPUs**, each with **at least 11G memory**

- Training

  `bash run_papers_base.sh train --work_dir PATH_TO_WORK_DIR`

- Evaluation

  `bash run_papers_base.sh eval --work_dir PATH_TO_WORK_DIR`
  
- If you have 2 gpus, you can still make it work with a simpler model with a little success tradeoff.

- Training

  `bash run_papers_base_2gpus.sh train --work_dir PATH_TO_WORK_DIR`

- Evaluation

  `bash rrun_papers_base_2gpus.sh eval --work_dir PATH_TO_WORK_DIR`

#### Other options:

- `--batch_chunk`: this option allows one to trade speed for memory. For `batch_chunk > 1`, the program will split each training batch into `batch_chunk` sub-batches and perform forward and backward on each sub-batch sequentially, with the gradient accumulated and divided by `batch_chunk`. Hence, the memory usage will propertionally lower while the computation time will inversely higher. 
- `--div_val`: when using adaptive softmax and embedding, the embedding dimension is divided by `div_val` from bin $i$ to bin $i+1$. This saves both GPU memory and the parameter budget.
- `--fp16` and `--dynamic-loss-scale`: Run in pseudo-fp16 mode (fp16 storage fp32 math) with dynamic loss scaling. 
  - Note: to explore the `--fp16` option, please make sure the `apex` package is installed (https://github.com/NVIDIA/apex/).
- To see performance without the recurrence mechanism, simply use `mem_len=0` in all your scripts.
- To see performance of a standard Transformer without relative positional encodings or recurrence mechanisms, use `attn_type=2` and `mem_len=0`.
