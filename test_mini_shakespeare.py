import numpy
from mingpt.utils import sample, fill_batch
import torch
import random
import copy

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
    model = torch.load('/home/fabian/control_gpt/trained_mini_sheaksspear')