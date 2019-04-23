import os
import shutil
import numpy as np
import pickle

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry


@registry.register_problem
class PaperGenerationProblem(text_problems.Text2SelfProblem):
    @property
    def corpus_url(self):
        return ("https://github.com/inzva/fake-academic-paper-generation/"
                "raw/master/dataset/preprocessed_data.txt")

    @property
    def is_generate_per_split(self):
        return False
    
    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    def _maybe_download_data(self, tmp_dir):
        if hasattr(self, "paper_dataset"):
            return self.paper_dataset
        else:
            generator_utils.maybe_download(tmp_dir, "paper_dataset.txt", self.corpus_url)
            paper_dataset_file = open(os.path.join(tmp_dir, "paper_dataset.txt"), 'rb')
            self.paper_dataset = paper_dataset_file.read().decode(encoding='utf-8')
            paper_dataset_file.close()
            return self.paper_dataset
    
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate samples of text.
        Args:
        data_dir: final data directory. Typically only used in this method to copy
            over user-supplied vocab files (for example, if vocab_type ==
            VocabType.TOKEN).
        tmp_dir: temporary directory that you can use for downloading and scratch.
        dataset_split: problem.DatasetSplit, which data split to generate samples
            for (for example, training and evaluation).
        Yields:
        Sample: dict<str feature_name, str text>: for language modeling problems
            (i.e. Text2SelfProblems), this generator should yield dicts with only
            the "targets" key.
        """
        paper_dataset = self._maybe_download_data(tmp_dir)

        data_seq_len = self.sequence_length - 1
        nb_samples = int(np.ceil(len(paper_dataset)/data_seq_len))
        for i in range(nb_samples):
            text = paper_dataset[i*data_seq_len : (i+1)*data_seq_len]
            yield {"targets": text}
    
    @property
    def sequence_length(self):
        """Length of each example (in tokens)."""
        return 128

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 100,
        }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
        }]