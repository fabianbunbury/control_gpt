
# you're on your own to define a class that returns individual examples as PyTorch LongTensors
from torch.utils.data import Dataset
import torch
from GPT import GPTmodel, GPT1Config, GPTConfig
import pdb


filename = "data/alice_in_wonderland.txt"

raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)

if __name__ == '__main__':
    vocab_size = 60
    horizon = 100

    second_config = GPT1Config(vocab_size , horizon)
    GPT_module = GPTmodel(second_config)
    basicTensor =  torch.ones(1,horizon, dtype=torch.int)
    pdb.set_trace()

    output =  GPT_module.forward(basicTensor)
    pdb.set_trace()
    




