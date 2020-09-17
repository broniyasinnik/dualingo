import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import MarianTokenizer, MarianMTModel
from typing import List, Union



class Marian(nn.Module):
    def __init__(self, src_lang: str, trg_lang: str):
        super(Marian, self).__init__()
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{trg_lang}'
        self.model = MarianMTModel.from_pretrained(model_name, normalize_embedding=True)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name, bos_token='<bos>', eos_token='<eos>')
        self._added_tokens = []

    def tokenize(self, sentences: List[str]):
        trgs = self.tokenizer(sentences, padding=True, return_tensors='pt')
        return trgs


    def add_tokens(self, new_tokens: List[str]):
        self._added_tokens.extend(new_tokens)
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def added_tokens_dict(self):
        ids = self.tokenizer.convert_tokens_to_ids(self._added_tokens)
        token_id_dict = dict(zip(self._added_tokens, ids))
        return token_id_dict


class CodesEmbeddings(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int,
                 weight: Tensor, codes_ids: List[int], wc_id: int):
        super(CodesEmbeddings, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                              _weight=weight)
        self.codes_ids = codes_ids
        self.wc_id = wc_id
        self._codes_weight: Tensor = None
    @property
    def codes_weight(self):
        return self._codes_weight.squeeze(1)

    @codes_weight.setter
    def codes_weight(self, weights: Tensor):
        self._codes_weight = weights.unsqueeze(1)

    def forward(self, input: Tensor):
        # Check if first token in input starts with weighted code
        if (input[:, 0] == self.wc_id).all():
            codes_tensor = torch.LongTensor(self.codes_ids)
            if torch.cuda.is_available():
                codes_tensor = codes_tensor.cuda()
            codes_tensor = codes_tensor.expand(input.size(0), -1)
            codes_input = torch.cat([codes_tensor, input[:, 1:]], dim=1)
            embedding = super(CodesEmbeddings, self).forward(codes_input)
            codes_embedding_part = embedding[:, :len(self.codes_ids), :]
            input_embedding_part = embedding[:, len(self.codes_ids):, :]
            weighted_part = self._codes_weight.matmul(codes_embedding_part)
            result_embedding = torch.cat([weighted_part, input_embedding_part], dim=1)
            return result_embedding
        else:
            embedding = super(CodesEmbeddings, self).forward(input)
            return embedding


class GumbelEncoder(nn.Module):
    def __init__(self, temp: float, categorical_dim: int):
        super(GumbelEncoder, self).__init__()

        self.categorical_dim = categorical_dim
        self.temp = temp

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, categorical_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return h2

    def forward(self, x, hard=False):
        q = self.encode(x)
        z = self.gumbel_softmax(q, hard)
        return z, F.softmax(q, dim=-1).reshape(*q.size())

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape)
        if torch.cuda.is_available():
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    @staticmethod
    def gumbel_softmax_sample(logits, temperature):
        y = logits + GumbelEncoder.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = GumbelEncoder.gumbel_softmax_sample(logits, self.temp)

        if not hard:
            return y.view(-1, self.categorical_dim)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, self.categorical_dim)


class BertForSentenceCodes(nn.Module):

    def __init__(self, num_sentence_codes: int):
        super(BertForSentenceCodes, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
                                                                   num_labels=num_sentence_codes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, sentences: List[str]):
        inputs = self.tokenizer(sentences, padding=True, return_tensors='pt')
        outputs, = self.model(**inputs)
        outputs = self.softmax(outputs)
        argmax = torch.argmax(outputs, dim=1)
        return argmax
