import os
import shutil
import numpy as np
import pickle

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder


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

        data_seq_len =  self.sequence_length
        self.nb_samples = int(np.ceil(len(paper_dataset)/data_seq_len))
        for i in range(self.nb_samples):
            text = paper_dataset[i*data_seq_len : (i+1)*data_seq_len]
            yield {"targets": text}

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        # override generate_encoded_samples, in order to override text2text_generate_encoded function
        if dataset_split == problem.DatasetSplit.TRAIN:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
        elif dataset_split == problem.DatasetSplit.EVAL:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        
        def text2text_generate_encoded(sample_generator,
                               vocab,
                               targets_vocab=None,
                               has_inputs=True,
                               inputs_prefix="",
                               targets_prefix=""):
            # override text2text_generate_encoded, in order to avoid EOS (end of string)
            # since for the problem, example sequences should not end 
            """Encode Text2Text samples from the generator with the vocab."""
            targets_vocab = targets_vocab or vocab
            for sample in sample_generator:
                if has_inputs:
                    sample["inputs"] = vocab.encode(inputs_prefix + sample["inputs"])
                    #sample["inputs"].append(text_encoder.EOS_ID)
                sample["targets"] = targets_vocab.encode(targets_prefix + sample["targets"])
                #sample["targets"].append(text_encoder.EOS_ID)
                yield sample
        
        return text2text_generate_encoded(generator, encoder,
                                        has_inputs=self.has_inputs,
                                        inputs_prefix=self.inputs_prefix,
                                        targets_prefix=self.targets_prefix)


