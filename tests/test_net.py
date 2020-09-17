import pytest
import torch
from model.net import Net,Net2
from utils import Params


@pytest.fixture
def src_lang_batch():
    batch = ['do you have animals?', 'i work as a professor.']
    return batch


@pytest.fixture
def trg_lang_batch():
    batch = ['neked van állatod?', 'professzorként dolgozom.']
    return batch

def test_net(src_lang_batch, trg_lang_batch):
    config = Params(src='en', trg='hu',
                    num_beams=3, max_length=20, num_return_sequences=3, cuda=False)
    model = Net(config)
    model(src_lang_batch, trg_lang_batch)
    return model

@pytest.fixture
def net2():
    config = Params(src='en', trg='hu', num_codes=1, gumbel_temp=0.5,
                    num_beams=3, max_length=20, cuda=False)
    model = Net2(config)
    return model

def test_debug_net(src_lang_batch, trg_lang_batch):
    config = Params(src='en', trg='hu', num_return_sequences=3,
                    num_beams=3, cuda=False)
    model = Net(config)
    # checkpoint = torch.load('../experiments/Net/runs/last.pth.tar',
    #                         map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['state_dict'])
    model(src_lang_batch, trg_lang_batch)
    # model.sample(src_lang_batch)

def test_debug_net2(src_lang_batch, trg_lang_batch):
    config = Params(src='en', trg='hu', num_codes=3, gumbel_temp=0.1,
                    num_beams=3, max_length=20, cuda=False)
    model = Net2(config)
    checkpoint = torch.load('../experiments/debug_model/runs/best.pth.tar',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model(src_lang_batch, trg_lang_batch)

def test_forward(net2, src_lang_batch, trg_lang_batch):
    net2.train()
    loss = net2(src_lang_batch, trg_lang_batch)
    assert False


def test_sample(net2, src_lang_batch):
    trg = net2.sample(src_lang_batch)
