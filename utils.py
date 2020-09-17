import csv
import functools
import hashlib
import itertools
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Set, Tuple
import unicodedata
import logging
import shutil
import torch

FIELDSEP = "|"


def makeID(text: str) -> str:
    """
    Create a unique ID based on the value of the input text.

    WARNING: This is typically used to create prompt IDs, but
    because of issues with stray spaces in the prompts,
    this may not always produce the ID you are expecting.
    """

    textID = hashlib.md5(text.lower().encode()).hexdigest()
    return f"prompt_{textID}"


def read_trans_prompts(lines: List[str]) -> List[Tuple[str, str]]:
    """
    This reads a file in the shared task format, returns a list of Tuples containing ID and text for each prompt.
    """

    ids_prompts = []
    first = True
    for line in lines:
        line = line.strip().lower()

        # in a group, the first one is the KEY. 
        # all others are part of the set. 
        if len(line) == 0:
            first = True
        else:
            if first:
                key, prompt = line.split(FIELDSEP)
                ids_prompts.append((key, prompt))
                first = False

    return ids_prompts


def strip_punctuation(text: str) -> str:
    """
    Remove punctuations of several languages, including Japanese.
    """
    return "".join(
        itertools.filterfalse(lambda x: unicodedata.category(x).startswith("P"), text)
    )


def read_transfile(lines: List[str], strip_punc=True, weighted=False) -> Dict[str, Dict[str, float]]:
    """
    This reads a file in the shared task format, and returns a dictionary with prompt IDs as 
    keys, and each key associated with a dictionary of responses. 
    """
    data = {}
    first = True
    options = {}
    key = ""
    for line in lines:
        line = line.strip().lower()

        # in a group, the first one is the KEY. 
        # all others are part of the set. 
        if len(line) == 0:
            first = True
            if len(key) > 0 and len(options) > 0:
                if key in data:
                    print(f"Warning: duplicate sentence! {key}")
                data[key] = options
                options = {}
        else:
            if first:
                key, _ = line.strip().split(FIELDSEP)
                first = False
            else:
                # allow that a line may have a number at the end specifying the weight that this element should take. 
                # this is controlled by the weighted argument.
                # gold is REQUIRED to have this weight.
                if weighted:
                    # get text
                    text, weight = line.strip().split(FIELDSEP)
                else:
                    text = line.strip()
                    weight = 1

                if strip_punc:
                    text = strip_punctuation(text)

                options[text] = float(weight)

    # check if there is still an element at the end.
    if len(options) > 0:
        data[key] = options

    return data


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, **kwargs):
        # check if the number of json configurations is 1
        self.__dict__.update(**kwargs)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class DataParams(Params):
    @classmethod
    def from_json(cls, data_json):
        assert os.path.exists(data_json), "The json path doesn't exists"
        self = cls()
        data_dir = os.path.dirname(data_json)
        self.dict['data_dir'] = data_dir
        self.update(data_json)
        return self


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_report(report, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        targets = report['targets']
        outputs = report['outputs']
        for prompt in targets:
            f.write(prompt + '\n')
            f.write("Expected Result:\n")
            for trg in targets[prompt]:
                f.write(trg + '\n')
            f.write("Output Result:\n")
            for out in outputs[prompt]:
                f.write(out + '\n')
            f.write('\n')


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class DataUtils:

    def __init__(self, data_dir):

        # Reading the dataset params from the json file
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = Params(json_path)
        self.src = self.dataset_params.src_lang
        self.trg = self.dataset_params.trg_lang
        self.bpe = '@@'
        self.bos = self.dataset_params.bos_word
        self.eos = self.dataset_params.eos_word

        # loading vocab (we require this to map words to their indices)
        vocab_path = os.path.join(data_dir, f'words_{self.src}.txt')
        self.vocab_src = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab_src[l] = i

        # setting the indices for UNKnown words and PADding symbols
        self.src_unk_ind = self.vocab_src[self.dataset_params.unk_word]
        self.src_pad_ind = self.vocab_src[self.dataset_params.pad_word]

        # loading tags (we require this to map tags to their indices)
        tags_path = os.path.join(data_dir, f'words_{self.trg}.txt')
        self.vocab_trg = {}
        with open(tags_path, encoding="utf-8") as f:
            for i, t in enumerate(f.read().splitlines()):
                self.vocab_trg[t] = i

        # Setting the indices for BOS and EOS
        self.trg_pad_ind = self.vocab_trg[self.dataset_params.pad_word]
        self.trg_bos_ind = self.vocab_trg[self.dataset_params.bos_word]
        self.trg_eos_ind = self.vocab_trg[self.dataset_params.eos_word]

    def get_src_vocab(self):
        '''
        This function returns the vocabulary of the source language
        the keys of the vocabulary are integers and the values are the words
        in the source language
        :return: Dictionary for the source language
        '''

        vocab = {i: word for word, i in self.vocab_src.items()}
        return vocab

    def get_trg_vocab(self):
        '''
        This function returns the vocabulary of the target language
        the keys of the vocabulary are integers and the values are the words
        in the target language
        :return: Dictionary for the target language
        '''
        vocab = {i: word for word, i in self.vocab_trg.items()}
        return vocab

    def detoknize_src_sen(self, sen):
        vocab = self.get_src_vocab()
        if type(sen) is torch.Tensor:
            detok_lst = [vocab[tok.item()] for tok in sen if tok.item() != self.src_pad_ind]
        else:
            detok_lst = [vocab[tok] for tok in sen if tok != self.src_pad_ind]

        detok_sen = ' '.join(detok_lst)
        detok_sen = detok_sen.replace(f'{self.bpe} ', '')
        return detok_lst, detok_sen

    def detokenize_trg_sen(self, sen):
        """
        Detokenize given sentence using the vocab
        :param sen: The sentence to dekoneize
        :param data_params: json file with the data set parameters
        :return:
        """
        vocab = self.get_trg_vocab()
        if type(sen) is torch.Tensor:
            detok_lst = [vocab[tok.item()] for tok in sen if tok.item() != self.trg_pad_ind]
        else:
            detok_lst = [vocab[tok] for tok in sen if tok != self.trg_pad_ind]

        detok_sen = ' '.join(detok_lst)
        detok_sen = detok_sen.replace(f'{self.bpe} ', '')
        if detok_sen.endswith(self.eos):
            detok_sen = detok_sen.replace(self.eos, '')
        if detok_sen.startswith(self.bos):
            detok_sen = detok_sen.replace(self.bos, '')

        return detok_lst, detok_sen.strip()
