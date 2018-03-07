import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F


# New stuff.
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

import numpy as np
import matplotlib.pyplot as plt
r = np.random.random([3000,2]) * 2 - 1.0
# Label points within a circle.
X = np.array([r[i] for i in range(r.shape[0]) if r[i,0] < -0.8 or r[i,0] > 0.8 ] )# and np.linalg.norm(r[i]) > 0.7])
plt.scatter(x =X[:, 0], y = X[:, 1])


LATENT_DIM = 8



# Compute the variational parameters for q
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(2, 200)
        self.linear2 = nn.Linear(200, LATENT_DIM)
        self.linear3 = nn.Linear(200, LATENT_DIM)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        return self.linear2(h), self.linear3(h)
    
    # Implement the generative model p(x | z)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(LATENT_DIM, 200)
        self.linear2 = nn.Linear(200, 2)

    def forward(self, z):
        return self.linear2(F.relu(self.linear1(z)))
    
    # VAE using reparameterization "rsample"

class NormalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(NormalVAE, self).__init__()

        # Parameters phi and computes variational parameters lambda
        self.encoder = encoder

        # Parameters theta, p(x | z)
        self.decoder = decoder
    
    def forward(self, x_src):
        # Example variational parameters lambda
        mu, logvar = self.encoder(x_src)
        
        q_normal = Normal(loc=mu, scale=logvar.mul(0.5).exp())

        # Reparameterized sample.
        z_sample = q_normal.rsample()
        #z_sample = mu
        return self.decoder(z_sample), q_normal
    
BATCH_SIZE = 32

mse_loss = nn.L1Loss(size_average=False)

# Problem setup.
encoder = Encoder()
decoder = Decoder()
vae = NormalVAE(encoder, decoder)

# SGD
learning_rate = 0.02
optim = torch.optim.SGD(vae.parameters(), lr = learning_rate)

NUM_EPOCHS = 50

# Get samples.
p = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM)), 
           V(torch.ones(BATCH_SIZE, LATENT_DIM)))

seed_distribution = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM)), 
                        V(torch.ones(BATCH_SIZE, LATENT_DIM)))
def graph_vae():
    fig, axs = plt.subplots(1,1)
    all = []
    all_out = []
    for k in range(500):
        seed =  seed_distribution.sample()
        x = decoder(seed[0:1] )
        all.append(x.data[0].numpy())
       
    all = np.array(all)
    axs.scatter(all[:, 0], all[:, 1])

graph_vae()


for epoch in range(NUM_EPOCHS):
    # Keep track of reconstruction loss and total kl
    total_loss = 0
    total_kl = 0
    total = 0
    alpha = 1
    for i, t in enumerate(X, BATCH_SIZE):
        if X[i:i+BATCH_SIZE].shape[0] < BATCH_SIZE : continue

        # Standard setup. 
        vae.zero_grad()
        x = V(torch.FloatTensor(X[i: i+BATCH_SIZE] ))

        # Run VAE. 
        out, q = vae(x)
        kl = kl_divergence(q, p).sum()

        # actual loss
        loss = mse_loss(out, x) + alpha * kl 
        loss = loss / BATCH_SIZE

        # record keeping.
        total_loss += mse_loss(out, x).data / BATCH_SIZE
        total_kl += kl.data / BATCH_SIZE
        total += 1
        loss.backward()
        optim.step()
    graph_vae()
    print(i, total_loss[0] / total , total_kl[0] / total)
    

z = V(torch.zeros(8))

all = []
for k in np.linspace(-3, 3, 30):
    seed = seed_distribution.sample()
    z[1] = k
    x = decoder(z)
    all.append(x.data.numpy())
all = np.array(all)
plt.scatter(all[:, 0], all[:, 1])