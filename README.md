# Neural Academic Paper Generation
inzva AI Projects #2 - Fake Academic Paper Generation Project

## Abstract

In this work, we tackle the problem of structured text generation, specifically academic paper generation in LaTeX, inspired by the surprisingly good results of basic character-level language models. Our motivation is using more recent and advanced methods of language modeling on a more complex dataset of LaTeX source files to generate realistic academic papers. Our first contribution is preparing a dataset with LaTeX source files on recent open-source computer vision papers. Our second contribution is experimenting with recent methods of language modeling and text generation such as Transformer and Transformer-XL to generate consistent LaTeX code. We report cross-entropy and bits-per-character (BPC) results of the trained models, and we also discuss interesting points on some examples of the generated LaTeX code. 

## Project Dependencies
- NumPy
- TexSoup (for dataset preparation)
- BeautifulSoup (for dataset preparation)
- Tensorflow 1.12 (for RNN)
- Tensor2Tensor 1.13.4 (for Transformer)
- PyTorch (for Transformer-XL)

## Dataset

*Note: We decided not to share the dataset because of ethical concerns. However, the code can be used to recreate the dataset.*

### Dataset Preparation
To the best of our knowledge there was no available dataset compiled from academic papers. Therefore we decided to prepare a dataset from academic papers on arxiv.org. 

All scripts related to the dataset preparation can be found in the **[dataset_generation](dataset_generation)** directory.

#### Steps for the dataset preparation:
##### 1) Select a subset of academic papers on arxiv.org  
We selected Computer Vision as the topic of interest for the dataset. Therefore, we crawled arxiv.org to find papers tagged as Computer Vision between 2015 - 2018. (BeautifulSoup is used as html parser)

related scripts: 
* **[dataset_generation/crawler.py](dataset_generation/crawler.py)** (crawles arxiv.org as specified and writes the result to **paperlinks.txt**)
* **[dataset_generation/random_paper_sampler.py](dataset_generation/random_paper_sampler.py)** (samples examples from **paperlinks.txt** and writes the result to **selected_papers.txt**)

##### 2) Download the source files
We downloaded the source files as tar files for the selected papers and untar/unzip them.

related script: **[dataset_generation/downloader.py](dataset_generation/downloader.py)** (reads selected papers from **selected_papers.txt**, downloads the source files and untar/unzip them)

##### 3) Find the latex source files for each paper and Compile each paper into one latex file
We resolved \include, \input kind of import statements in latex source files in order to compile each paper into one latex file and wrote a latex file for each paper. 

related script: **[dataset_generation/latex_input_resolver.py](dataset_generation/latex_input_resolver.py)** (Finds the latex files from the source files, reads the content using TexSoup, finds the root files(files including documentclass statement), recursively replaces the import statements with the content of the imported file, and writes a latex file for each paper.)

##### other helper scripts:
* **[dataset_generation/complete_dataset.py](dataset_generation/complete_dataset.py)** (kind of combination of all these scripts which finds problematic source files and replaces them with other papers from the **paperlinks.txt**)
* **[dataset_generation/renumber_paper.py](dataset_generation/renumber_paper.py)** (renames the papers like 0.tex, 1.tex, 2.tex so on)

Using this specified process, we downloaded 4-5 GB source files for papers since source files include images etc. which are not need for our purpose. At the end, we have 799 latex files each for an academic paper. Before preprocessing, this is approximately equal to 46 MB of latex. 

### Preprocessing
Dataset is needed to be preprocessed because of noise such as created by comments and non-UTF characters. Therefore, we used _preprocess_char.py_ to delete comments and characters that used below a certain threshold, in our experiments it is 100. 

For our baseline model, we decided to use character level embedding. The details of the preprocessed char-based dataset is given below.

|         **Feature**         |  **Value** |
|:------------------------------:|:----------:|
| Number of Unique Token              |     102    |
| Number of Token                     | 37,921,928 |
| Lower-case to Upper-case Ratio |    23.95   |
| Word to Non-word Ratio         |    3.17    |

## Models

### Baseline Model (RNN)
The rnn model described in the blog post "The Unreasonable Effectiveness of Recurrent Neural Networks"[1] 
#### How to Run:
After preparing the dataset, run **[char-rnn.py](char-rnn.py)** to train the model.

When training is over, run **[generate_text.py](generate_text.py)**. This script will load the last
checkpoint and generate a number of characters using the learned parameters.

### Transformer
Transformer [2] is another popular model.

#### How to Run:
We use [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) [3] for Transformer model. See [t2t_paper_generation_problem](t2t_paper_generation_problem) directory for details.

### Transformer-XL
Transformer-XL [4] is a new model aiming to extend Transformer such that long term dependecies could be handled properly.

#### How to Run:
We use the original code shared by the authors who propose Transformer-XL. See [transformer-xl](transformer-xl) directory for details.

## References

[1] The Unreasonable Effectiveness of Recurrent Neural Networks
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

[2] Vaswani, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.

[3] Vaswani et al. "Tensor2Tensor for Neural Machine Translation". 2018. [arXiv:1803.07416](http://arxiv.org/abs/1803.07416)

[4] Dai et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context". 2018. [arXiv:1901.02860](http://arxiv.org/abs/1901.02860)