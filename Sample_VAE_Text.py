import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F


# New stuff.
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

import numpy as np
import matplotlib.pyplot as plt

import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import GloVe

TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)

# make splits for data
train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100))
LABEL.build_vocab(train)

WORD_DIM = 100
LATENT_DIM = 32
num_embeddings = len(TEXT.vocab)

# LSTM Encoder / Inference Network
class Encoder(nn.Module):
    def __init__(self, num_embeddings):
        super(Encoder, self).__init__()
        self.emb_layer = nn.Embedding(embedding_dim=WORD_DIM, 
                                      num_embeddings=num_embeddings)
        self.emb_layer.weight.data = TEXT.vocab.vectors.clone()
        self.enc_layer = nn.LSTM(input_size=WORD_DIM, hidden_size=WORD_DIM)
        self.mu_layer = nn.Linear(WORD_DIM, LATENT_DIM)
        self.logvar_layer = nn.Linear(WORD_DIM, LATENT_DIM)
        
    def forward(self, src):
        emb = self.emb_layer(src)
        output, _  = self.enc_layer(emb)
        final = output[:, -1]
        mu = self.mu_layer(final)
        logvar = self.logvar_layer(final)
        return mu, logvar
    
# Bag-of-Word Generative Model
class Decoder(nn.Module):
    def __init__(self, num_embeddings):
        super(Decoder, self).__init__()
        self.layer1 = nn.Linear(LATENT_DIM, 200)
        self.layer2 = nn.Linear(200, num_embeddings)
    def forward(self, hidden):
        return self.layer2(F.relu(self.layer1(hidden)))
    
BATCH_SIZE = 32
train_iter, test_iter = data.BucketIterator.splits(
    (train, test), batch_size=BATCH_SIZE, device=-1)
bce = nn.BCEWithLogitsLoss(size_average=False)
encoder = Encoder(len(TEXT.vocab))
decoder = Decoder(len(TEXT.vocab))
vae = NormalVAE(encoder, decoder)
learning_rate = 0.01
optim = torch.optim.SGD(vae.parameters(), lr = learning_rate)
#optim = torch.optim.Adam(vae.parameters(), lr = learning_rate)
NUM_EPOCHS = 50
p = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM)), V(torch.ones(BATCH_SIZE, LATENT_DIM)))

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_kl = 0
    total = 0
    alpha = 1
    for i, t in enumerate(train_iter):
        if t.label.size(0) != BATCH_SIZE : continue
        vae.zero_grad()
        x, _ = t.text
        out, q = vae(x)
        kl = kl_divergence(q, p).sum()
        target = torch.zeros(BATCH_SIZE, len(TEXT.vocab)).float()
        for b in range(BATCH_SIZE):
            target[b][t.text[0].data[b]] =  1   
        loss = bce(out.view(-1), V(target.view(-1) ) ) + alpha * kl 
        loss = loss / BATCH_SIZE
        
        total_loss += bce(out.view(-1), V(target.view(-1) ) ).data / BATCH_SIZE
        total_kl += kl.data / BATCH_SIZE
        total += 1
        loss.backward()
        optim.step()
    print(i, total_loss[0] / total , total_kl[0] / total)