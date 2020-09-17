import torch
from torch import Tensor
import copy
import torch.nn as nn
import utils
import functools
from typing import List, Union, Dict
from .networks import BertForSentenceCodes, Marian, GumbelEncoder, CodesEmbeddings
from transformers import MarianTokenizer, MarianMTModel


class Net(nn.Module):
    def __init__(self, config: utils.Params):
        # the embedding takes as input the vocab_size and the embedding_dim
        super(Net, self).__init__()
        model_name = f'Helsinki-NLP/opus-mt-{config.src}-{config.trg}'
        self.config = config
        self.model = MarianMTModel.from_pretrained(model_name, normalize_embedding=True, num_beams=10)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"bos_token": '<bos>'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def add_special_tokens(self, num_additional_tokens: int):
        special_tokens_dict = dict()
        additional_tokens_lst = [f'<cls{i}>' for i in range(num_additional_tokens + 1)]
        special_tokens_dict['additional_special_tokens'] = additional_tokens_lst
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def encode_src_lang(self, src: List[str]):
        src_encoded = self.tokenizer.batch_encode_plus(src,
                                                       pad_to_max_length=True,
                                                       return_tensors='pt')
        input_ids = src_encoded['input_ids']
        attention_mask = src_encoded['attention_mask']
        return input_ids, attention_mask

    def encode_trg_lang(self, trgs: List[str]):
        trgs = [self.tokenizer.bos_token + trg for trg in trgs]
        trg_encoded = self.tokenizer.batch_encode_plus(trgs,
                                                       pad_to_max_length=True,
                                                       return_tensors='pt')
        input_ids = trg_encoded['input_ids']
        attention_mask = trg_encoded['attention_mask']
        return input_ids, attention_mask

    def decode_trg_lang(self, trgs: List[List[int]]):
        valid_trgs = []
        # truncate until the eos
        # for trg in trgs:
        #     eos_token_id = self.tokenizer.eos_token_id
        #     valid_tokens_ids = slice(trg.index(eos_token_id) if eos_token_id in trg else -1)
        #     valid_tokens = trg[valid_tokens_ids]
        #     valid_trgs.append(valid_tokens)

        detok = self.tokenizer.batch_decode(trgs,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
        SPIECE_UNDERLINE = "▁"
        detok = [trg.replace(SPIECE_UNDERLINE, ' ').strip() for trg in detok]
        return detok

    def forward(self, src_lang_batch: List[str], trg_lang_batch: List[str]):
        trg_lang_batch = [self.tokenizer.bos_token+trg for trg in trg_lang_batch]
        inputs_batch = self.tokenizer.prepare_seq2seq_batch(src_lang_batch,trg_lang_batch, return_tensors='pt')
        input_ids = inputs_batch['input_ids']
        attention_mask = inputs_batch['attention_mask']
        decoder_input_ids = inputs_batch['labels'][:,:-1]
        labels = inputs_batch['labels'][:,1:].clone().detach()

        if self.config.cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            decoder_input_ids = decoder_input_ids.cuda()
            labels = labels.cuda()

        outputs = self.model(input_ids=input_ids,
                             decoder_input_ids=decoder_input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        loss = outputs[0]
        return loss

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample(self, src_batch: List[str]):
        if type(src_batch) is str:
            src_batch = [src_batch]
        batch = self.tokenizer.prepare_seq2seq_batch(src_texts=src_batch)  # don't need tgt_text for inference
        input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
        attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
        gen = self.model.generate(input_ids,
                                  num_beams=self.config.num_beams,
                                  decoder_start_token_id=self.tokenizer.bos_token_id,
                                  max_length=25,
                                  early_stopping=True,
                                  # use_cache=False,
                                  # no_repeat_ngram_size=1,
                                  eos_token_id=self.tokenizer.eos_token_id,
                                  attention_mask=attention_mask,
                                  num_return_sequences=self.config.num_return_sequences)
        decode_gen = self.decode_trg_lang(gen)
        return decode_gen


class Net2(nn.Module):
    def __init__(self, config: utils.Params):
        super(Net2, self).__init__()
        self.config = config
        self.marian = Marian(src_lang=config.src, trg_lang=config.trg)
        self.codes = [f'<C{i}>' for i in range(config.num_codes)]
        self.cw = '<CW>'
        self.marian.add_tokens(self.codes + [self.cw])
        self.encoder = self.marian.model.get_encoder()
        self.encoder2 = copy.deepcopy(self.encoder)
        self.gumbel = GumbelEncoder(config.gumbel_temp, config.num_codes)
        marian_embed = self.marian.model.get_input_embeddings()
        cw_id = self.marian.added_tokens_dict.get(self.cw)
        codes_ids = [self.marian.added_tokens_dict.get(code) for code in self.codes]
        self.codes_embed = CodesEmbeddings(marian_embed.num_embeddings,
                                           marian_embed.embedding_dim,
                                           marian_embed.padding_idx,
                                           marian_embed.weight,
                                           codes_ids=codes_ids,
                                           wc_id=cw_id)
        self.marian.model.set_input_embeddings(self.codes_embed)

    @staticmethod
    def _prefix_with_codes_ids(input_ids: torch.LongTensor, codes_ids: Union[int, List[int]]):
        if type(codes_ids) is int:
            codes_ids = [codes_ids]
        codes = torch.LongTensor(codes_ids)
        codes = codes.expand(input_ids.size(0), -1)
        input_ids = torch.cat([codes, input_ids], dim=1)
        return input_ids

    def targets_decoding(self, targets: Tensor) -> List[str]:
        targets = self.marian.tokenizer.batch_decode(targets[:, 1:], skip_special_tokens=True)
        SPIECE_UNDERLINE = "▁"
        targets = [trg.replace(SPIECE_UNDERLINE, ' ').strip() for trg in targets]
        return targets

    def prepare_training_batch(self, src_lang_batch: List[str], trg_lang_batch: List[str]):
        encoder_src_inputs = self.marian.tokenize(src_lang_batch)
        encoder_trg_inputs = self.marian.tokenize(trg_lang_batch)
        trg_lang_batch = ['<CW>' + trg for trg in trg_lang_batch]
        decoder_inputs = self.marian.tokenize(trg_lang_batch)
        if self.config.cuda:
            encoder_src_inputs = {k: v.cuda() for k, v in encoder_src_inputs.items()}
            encoder_trg_inputs = {k: v.cuda() for k, v in encoder_trg_inputs.items()}
            decoder_inputs = {k: v.cuda() for k, v in decoder_inputs.items()}
        return encoder_src_inputs, encoder_trg_inputs, decoder_inputs

    def forward(self, src_lang_batch: List[str], trg_lang_batch: List[str]):
        kl_loss = 0
        if self.training:
            encoder_src_inputs, encoder_trg_inputs, decoder_inputs = self.prepare_training_batch(src_lang_batch, trg_lang_batch)
            encoder_src_outputs = self.encoder(**encoder_src_inputs)
            encoder_trg_outputs = self.encoder2(**encoder_trg_inputs)
            encoder_trg_outputs_mean = torch.mean(encoder_trg_outputs[0], dim=1)
            weights, posterior = self.gumbel(encoder_trg_outputs_mean, hard=True)
            self.codes_embed.codes_weight = weights
            log_ratio = torch.log(posterior * len(self.codes) + 1e-20)
            kl_loss = torch.sum(posterior * log_ratio, dim=-1).mean()

        lm_loss, prediction_scores, _ = self.marian.model(encoder_src_inputs['input_ids'],
                                                          encoder_outputs=encoder_src_outputs,
                                                          decoder_input_ids=decoder_inputs['input_ids'][:, :-1],
                                                          labels=decoder_inputs['input_ids'][:, 1:].clone())

        return lm_loss + kl_loss

    def _sample_targets_by_code(self, src_sentences: List[str]):
        encoder_inputs = self.marian.tokenize(src_sentences)
        if self.config.cuda:
            encoder_inputs = {k: v.cuda() for k, v in encoder_inputs.items()}
        sampeled_target_by_code: Dict[str: List[str]] = {code: [] for code in self.codes}
        for code in self.codes:
            start_code_id = self.marian.added_tokens_dict[code]
            targets = self.marian.model.generate(encoder_inputs['input_ids'],
                                                 num_beams=self.config.num_beams,
                                                 decoder_start_token_id=start_code_id,
                                                 max_length=self.config.max_length,
                                                 early_stopping=True,
                                                 num_return_sequences=self.config.num_beams)

            sampeled_target_by_code[code] = self.targets_decoding(targets)
        return sampeled_target_by_code

    def sample(self, src_sentences: List[str]):
        target_by_code_dict = self._sample_targets_by_code(src_sentences)
        all_targets = functools.reduce(lambda a, b: a + b, target_by_code_dict.values())
        return all_targets