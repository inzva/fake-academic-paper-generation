# Fake Academic Paper Generation Project
inzva AI Projects #2 - Fake Academic Paper Generation Project

## Project Description

In this project, we aim to use the LaTeX source files of open access papers on arXiv
as a dataset and feed it into a neural network to be able to generate realistic
looking academic papers. We chose the character based recurrent neural network (RNN)
model used by Andrej Karpathy in his blog post as our baseline [1]. We will try to improve
the baseline results of the char-RNN model by applying transformers and attention
mechanism [2]. We also want to try GANs to generate realistic LaTeX code. [3]

## Dataset

To be explained

## Project Dependencies

- Tensorflow 1.12
- NumPy
- TexSoup (for dataset preparation)
- BeautifulSoup (for dataset preparation)

## How to Run

##### Baseline Model
After preparing the dataset, run **char-rnn.py** to train the model.

When training is over, run **generate_text.py**. This script will load the last
checkpoint and generate a number of characters using the learned parameters.

## References

[1] The Unreasonable Effectiveness of Recurrent Neural Networks
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

[2] Vaswani, Ashish, et al. "Attention is all you need." *Advances in Neural Information Processing Systems*. 2017.

[3] Nie, Weili, Nina Narodytska, and Ankit Patel. "RelGAN: Relational Generative Adversarial Networks for Text Generation." (2018).