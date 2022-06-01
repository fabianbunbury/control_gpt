import numpy
from mingpt.utils import sample, fill_batch
import torch
import random
import copy
import pdb
from torch.utils.data import Dataset
import os
from torch.nn import functional as F
import numpy as np

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out
class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        chars.append('_blanc_')
        data_size, vocab_size = len(data), len(chars) # add one to the number of chars to account for blanks

        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size


    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """

        y = torch.tensor(dix[:-1], dtype=torch.long)
        x = copy.deepcopy(y)

        ## randomly delete blocks of info in the data set
        self.num_state_types = 1
        self.num_states_per_step = 1


        deletion_mode = random.randint(0,0)
        if deletion_mode == 0: #this mode deletes all elements at a certain time

            start_deleting_from = random.randint(0,(self.block_size/self.num_states_per_step)-1)

            stop_deleting_from = random.randint(start_deleting_from, (self.block_size/self.num_states_per_step)-1)
            index_of_states_to_delete = torch.range(start_deleting_from*self.num_states_per_step, 
                                                    stop_deleting_from*self.num_states_per_step,    
                                                    1 )
            x[index_of_states_to_delete.numpy()] = self.vocab_size-1


        elif deletion_mode == 1:

            index_of_patches_to_delete = 2
            """ the deleted blocks should not be entierly random but
            instead should be large contiguous blocks. """
        return x, y


if __name__ == '__main__':
    block_size = 128 # spatial extent of the model for its context
    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    test_dataset = CharDataset(text, block_size)
    pdb.set_trace()
    path = os.path.dirname(os.path.abspath(__file__))+'/trained_mini_sheaksspear'

    model = torch.load(path)
    input_data, output_data = test_dataset.__getitem__(0)
    # x = torch.tensor([test_dataset.stoi[s] for s in input_data], dtype=torch.long)[None,...]
    x = input_data[None,...]
    x = x.to(torch.device("cuda"))
    logits, _ = model(x) 
    temperature = 1.0
    logits = logits[:, :, :] / temperature
    top_k = 10
    logits_topk, indexes = torch.topk(logits, top_k)

    most_likely =  indexes[:,:,0]
    logits[0,np.arange(0,10),np.arange(4,14)]
    probs = F.softmax(logits_topk, dim=-1)
    # ix = torch.multinomial(probs, num_samples=1)
    _, ix = torch.topk(probs, k=1, dim=-1)

    input_data = ''.join([test_dataset.itos[int(i)] for i in input_data])
    print('\n #########input_data##########')
    print(input_data)
    text_output_from_transformer = ''.join([test_dataset.itos[int(i)] for i in most_likely[0]])
    print('\n #############text_output_from_transformer##############')
    print(text_output_from_transformer)
    target_data = ''.join([test_dataset.itos[int(i)] for i in output_data])
    print('\n ############target_data###############')
    print(target_data)
    sampled_data = torch.multinomial(probs[0],num_samples=1).cpu().detach().numpy()

    # indexes[sampled_data]
    indexes_numpy = indexes[0].cpu().detach().numpy()
    # np.take(indexes_numpy,sampled_data.flatten(),-1)
    # indexes_numpy[0,:,sampled_data.flatten()]
    index_of_elements = sampled_data.flatten()
    dim_0_dimension_of_index = np.arange(0,len(index_of_elements))
    # index_of_indexes = np.stack((dim_0_dimension_of_index,index_of_elements),1).tolist()
    # indexes_numpy[index_of_indexes]
    sampled_values = indexes_numpy[dim_0_dimension_of_index,index_of_elements]
    target_data = ''.join([test_dataset.itos[int(i)] for i in sampled_values])
    print('\n ############sampled_data###############')
    print(target_data)
    test = [[0,1],[2,2]]
    


  

