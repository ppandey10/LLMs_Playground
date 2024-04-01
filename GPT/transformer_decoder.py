# import necessary libraries
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#=======================================================================================#
# Setup the hyperparameters
batch_size = 16 # ho of independent sequences that will processed in parallel
block_size = 32 # context length
max_iters = 10000
eval_interval = 100
learning_rate = 1e-4
eval_iters = 200
n_embd = 64
num_of_heads = 4
dropout = 0.2

#========================================================================================#
# Load and read the text file
with open('/Net/Groups/BGI/scratch/ppandey/LLMs_Playground/The_Great_Gatsby.txt', 'r', encoding='utf-8') as file:
    txt_file = file.read()

# Get unique characters
chars = sorted(set(txt_file))
# Develop the vocab_size
vocabulary_size = len(chars)

#========================================================================================#
# Encoder and decoder for characterwise tokenisation
string_to_int = {
    ch:i for i,ch in enumerate(chars)
}

int_to_string = {
    i:ch for i,ch in enumerate(chars)
}

charwise_encoder = lambda input_word: [string_to_int[char] for char in input_word]
charwise_decoder = lambda input_tokenised_word: ''.join([int_to_string[i] for i in input_tokenised_word])

#=======================================================================================#
# Create the dataset
data = torch.tensor(charwise_encoder(txt_file), dtype=torch.long)

# Split the dataset
train_percentage_split = int(0.9*len(data))
train_data = data[:train_percentage_split] # 90% training 
val_data = data[train_percentage_split:]

#=======================================================================================#
# Create the dataloader 
def get_batch(split):
    data = train_data if split=='train' else val_data
    random_indexes = torch.randint(0, (len(data) - block_size), (batch_size,))
    # print(random_indexes)
    x = torch.stack([data[i:i+block_size] for i in random_indexes])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_indexes])
    # move the data to gpus
    x, y = x.to(device), y.to(device)
    return x, y

#=======================================================================================#
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # good practice: keep in mind the mode we are in. this helps us understand/build our model correctly. 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#======================================================================================#
# Create single head class
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Since, we want masked multi-head attention
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # create key vector for each token
        q = self.query(x) # create query vector for each token 
        # Calculating attention score
        wei = q @ k.transpose(-2,-1) * C**-0.5
        # Masking
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        # Apply softmax i.e. normalisation
        wei = F.softmax(wei, dim=-1)
        # Apply dropout
        wei = self.dropout(wei)
        # Get value
        v = self.value(x)
        # Multiply with the wei
        out = wei @ v
        return out 
    
#======================================================================================#
# Create multi-head attention
class MultiHeadAttention(nn.Module):
    '''Parallel processing of multiple heads'''

    def __init__(self, num_of_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_of_heads)])
        # For stable training
        self.proj = nn.Linear(n_embd, n_embd) # output channel = embedding size 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Concatenate the result from each head
        out = torch.cat([h(x) for h in self.heads], dim=-1) # each head will gegt concatenated along channel dim = embedding size
        out = self.dropout(self.proj(out)) 
        return out

#======================================================================================#
# Create a feed forward block
class FeedForwardBlock(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
#======================================================================================#
class TransformerBlock(nn.Module):

    def __init__(self, n_embd, num_of_heads):
        super().__init__()
        head_size = n_embd // num_of_heads
        self.sa = MultiHeadAttention(num_of_heads, head_size)
        self.ffwd = FeedForwardBlock(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.layer_norm_1(x)) # residual connection
        x = x + self.ffwd(self.layer_norm_2(x))
        return x
    
#======================================================================================#
# Create the Tranformer decoder
class GPTDecoderModel(nn.Module):
    '''Simple transformer decoder model'''

    def __init__(self):
        super().__init__()
        # start by builing the token embedding table
        self.token_embd_table = nn.Embedding(vocabulary_size, n_embd)
        # then we move to position embedding
        self.position_embd_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            TransformerBlock(n_embd, num_of_heads),
            TransformerBlock(n_embd, num_of_heads),
            TransformerBlock(n_embd, num_of_heads),
            TransformerBlock(n_embd, num_of_heads),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

    def forward(self, index, targ=None):
        B, T = index.shape
        token_embd = self.token_embd_table(index)
        pos_embd = self.position_embd_table(torch.arange(0, T, device=device))
        x = token_embd + pos_embd
        x = self.blocks(x)
        # Create logits
        logits = self.lm_head(x)

        if targ is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targ.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # Predictions
    def generate(self, index, max_number_tokens):
        for _ in range(max_number_tokens):
            # Since, we train on a limited block size 
            index_cond = index[:,-block_size:]
            # Get predicitons
            logits, loss = self.forward(index_cond)
            # consider only last time step 
            logits = logits[:, -1, :]
            # softmax for probabilities
            probs = F.softmax(logits, dim=-1)
            # getting distribution
            next_index = torch.multinomial(probs, num_samples=1)
            # concatenate the predictions with the previous output
            index = torch.cat((index, next_index), dim=1)
        return index
    
#======================================================================================#
# Training
model = GPTDecoderModel() # vocabulary_size=88 (The_Great_Gatsby)
m = model.to(device)

# start by defining an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loss_progress = []

for i in tqdm(range(max_iters), desc="loss:"):

    if i % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {i}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # sample a batch of data 
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#======================================================================================#
# Lets do some predictions
# Start by feeding some input (get some index)
idx = torch.zeros((1,1), dtype=torch.long, device=device)
generate_values = m.generate(index=idx, max_number_tokens=500)
print(charwise_decoder(generate_values[0].tolist()))



        



