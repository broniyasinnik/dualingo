import pytest
import torch
from model.networks import CodesEmbeddings, Marian


@pytest.fixture
def marian():
    marian = Marian(src_lang='en', trg_lang='hu')
    return marian


@pytest.fixture
def target_batch(marian):
    targets = ["megfőzi az ételt.", "a harmadik alma vörös."]
    return marian.tokenize(targets)


@pytest.fixture
def codes_embeddings(marian):
    codes = ['<C1>', '<C2>', '<C3>']
    wc_code = ['<WC>']
    marian.add_tokens(codes + wc_code)
    codes_ids = marian.tokenizer.convert_tokens_to_ids(codes)
    wc_id = marian.tokenizer.convert_tokens_to_ids(wc_code)[0]
    marian_embed = marian.model.get_input_embeddings()
    codes_embed = CodesEmbeddings(marian_embed.num_embeddings, marian_embed.embedding_dim,
                                  padding_idx=marian_embed.padding_idx,
                                  weight=marian_embed.weight,
                                  codes_ids=codes_ids, wc_id=wc_id)
    return codes_embed


def test_forward(codes_embeddings, target_batch):
    input_ids = target_batch['input_ids']
    wc_tensor = torch.full((input_ids.size(0), 1),
                           codes_embeddings.wc_id,
                           dtype=input_ids.dtype)
    wc_input = torch.cat([wc_tensor, input_ids], dim=1)
    batch_size, num_codes = input_ids.size(0), len(codes_embeddings.codes_ids)
    codes_weights = torch.randn((batch_size, num_codes))
    codes_embeddings.codes_weight = codes_weights
    result_embedding = codes_embeddings(wc_input)
    assert result_embedding.size() == wc_input.size()



