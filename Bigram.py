########################################################################
### Generate Shakespeare via a BiGram Character-level Language Model ###
########################################################################

## Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(196)

#######################
### Hyperparameters ###
#######################

context_len=8
batch_size=4

### Remember our input: 1D array of length context_len. The length of the context. the indicies of the context in our vocab! 
### e.g. [10, 8, 9, 0, 1, 5, 7]

#############
## Dataset ##
#############

with open('tiny_shakespeare.txt', 'r') as file:
    text=file.read()

# Create Vocabulary (over characters)
vocab=sorted(list(set(text)))
vocab_size=len(vocab)

#Create Vocab Mappings
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}

# create function to encode text strings and decode indicies
def encode(text:str):
    return [char_to_idx[char] for char in text]
def decode(indices: list):
    return ''.join([idx_to_char[idx] for idx in indices])

#encode entire dataset. Stored as a tensor.
data = torch.tensor(encode(text), dtype=torch.int64)

#######################################
### Data Pre-processing and Loading ###
#######################################

#Split into training and validation
split = int(data.shape[0]*0.9) #90% Train 10% Validation
train_data = data[:split]
val_data = data[split:]

def create_batch(split:str):
    """
    Create batches of data by randomly sampling windows of length context_len
    from the dataset
    
    split | str: "train" or "val. Defines which dataset to create batches from"
    """
    data: torch.tensor = train_data if split=="train" else val_data
    
    # randomly select batch_size number of starting points in the data
    # The upper bound is (len(data) - context_len) to ensure we can always build a complete
    # context of size context_len, even if we start at the last valid position
    starting_idxs: torch.tensor = torch.randint(0, len(data)-context_len, size=(batch_size,))
    
    #create a list of tensors, which we then stack
    X=torch.stack([data[start:start+context_len] for start in starting_idxs])
    y=torch.stack([data[start+1: start+context_len+1] for start in starting_idxs])
    
    return X, y



#############################
### BiGram Language Model ###
#############################

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        """
        Define Layers and Parameters
        """
        super().__init__()
        self.bigram_lookup_table = nn.Embedding(vocab_size, vocab_size)
    
    
    def forward(self, idxs, targets=None):
        """
        Define how data flows through the layers of the network
        
        idxs | (batch_dim, context_len): batches of indices of characters in our Vocab. 
        targets | (batch_dim, context_len): batch of target indices
        """
        # For each index in idxs you'll have a 1D array of length vocab_size representing the "frequency" dist over the whole vocab given that index
        logits = self.bigram_lookup_table(idxs) # (batch_dim, context_len, vocab_size) 
        
        if targets == None: # we are generating
            loss=None
        else:
            ## ----- Loss ----- ##
            # ve- log likelihood <==> Cross entropy in this senario
            # Cross Entropy wants the last dim to be num_classes (i.e. the len(vocab))
            # reshape/view
            batch_size, context_len, num_classes = logits.shape
            logits = logits.view(batch_size*context_len, num_classes) # stack the batches. each row represents one example of context
            targets = targets.view(batch_size*context_len) # long 1D array
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idxs, max_token_length):
        """
        Define how the model goes about generating new tokens
        
        idxs | (batch_dim, context_len): batches of indices of characters in our Vocab.
        max_token_length | int: Maximum number of tokens to generate
        """
        # No longer generating names. Now we want to, given a sequence (idxs), generate new tokens and concat to idxs.
        
        ## Generate token by token
        for i in range(max_token_length):
            
            # Calling self() or model() in Pytorch calls forward()
            logits, _ = self(idxs, targets=None)
            
            # we only need the one context (as Bigram model). logits shape: (batch_dim, context_len, vocab_len)
            # take the last context in each batch. So one row is taken from each batch giving shape (batch_dim, vocab_len)
            logits = logits[:, -1, :] 
            
            ## create prob_dists and sample ##
            prob_dists = torch.softmax(logits, dim=1) # across the columns (so dim=1). Very Easy when you understand!
            indicies = torch.multinomial(prob_dists, num_samples=1, replacement=True) # (batch_dim, 1) out of the vocab_len one is chosen
            
            ### KEY: we append the generated indicies onto orginal idxs. Then loop back to generate again!
            idxs = torch.cat((idxs, indicies), dim=1)
        return idxs
    

############
### Main ###
############


bigram = BigramLanguageModel(vocab_size)
optimiser = torch.optim.AdamW(bigram.parameters(), lr=1e-3)
