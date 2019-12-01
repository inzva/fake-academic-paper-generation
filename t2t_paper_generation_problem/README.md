## Tensor2Tensor Paper Generation Problem

### Install Dependecies
```
pip install tensor2tensor==1.13.2
```

### Train Model
**First, you need to change [PaperGenerationProblem.py](PaperGenerationProblem.py) line 20 with the URL to the dataset created using the code specied in [the main README](../README.md).**

```
python t2t_paper_generation_problem/train.py
```

```
usage: train.py [-h] [--folder FOLDER] [--model MODEL]
                [--hparams_set HPARAMS_SET]

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER
  --model MODEL
  --hparams_set HPARAMS_SET
```

# Generate Paper from the Trained Model
See [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)