import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F


# New stuff.
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

import numpy as np
import matplotlib.pyplot as plt

r = np.random.random([1000,2]) * 2 - 1.0
# Label points within a circle.
X = np.array([r[i] for i in range(r.shape[0]) if np.linalg.norm(r[i]) > 0.5 ])
plt.scatter(x =X[:, 0], y = X[:, 1])

LATENT_DIM = 32
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(LATENT_DIM, 100)
        self.linear2 = nn.Linear(100, 2)
    def forward(self, z):
        return self.linear2(F.relu(self.linear1(z)))
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, point):
        return F.sigmoid(self.linear2(F.relu(self.linear1(point))))

G = Generator()
D = Discriminator()
learning_rate = 0.01
optim_gen = torch.optim.SGD(G.parameters(), lr=learning_rate)
optim_disc = torch.optim.SGD(D.parameters(), lr=learning_rate)
seed_distribution = Normal(V(torch.zeros(BATCH_SIZE, LATENT_DIM)), 
                           V(torch.ones(BATCH_SIZE, LATENT_DIM)))

def graph():
    fig, axs = plt.subplots(1,1)
    all = []
    all_out = []
    for k in range(500):
        seed = seed_distribution.sample()
        x_fake = G(seed[0:1])
        out = D(x_fake)        
        all.append(x_fake.data[0].numpy())
        all_out.append(out)
    all = np.array(all)
    out = np.array(all_out)
    axs.scatter(all[:, 0], all[:, 1], color=["blue" if c < 0.5 else "red" for c in out])

for epoch in range(50):
    total_gen_loss = 0
    total_disc_loss = 0
    for i, t in enumerate(X, BATCH_SIZE):
        if X[i:i+BATCH_SIZE].shape[0] < BATCH_SIZE : continue

        # Grad real
        # -E[log(D(x))]
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        x_real = V(torch.FloatTensor(X[i:i+BATCH_SIZE]))
        d = D(x_real)
        loss_a = 0.5 * -d.log().mean()
        loss_a.backward()
        
        # Grad fake
        # -E[log(1 - D(G(z)) )]
        seed = seed_distribution.sample()
        x_fake = G(seed)
        d = D(x_fake.detach())        
        loss_b = 0.5 * -(1 - d + 1e-10).log().mean()
        loss_b.backward()
        optim_disc.step()
        total_disc_loss += loss_a.data[0] + loss_b.data[0]

        # Grad generator
        # E[log(1 - D(G(z)))]
        optim_disc.zero_grad()
        # No detach here.
        d = D(x_fake)
        loss_c = (1 - d + 1e-10).log().mean()
        #loss_c = -(d + 1e-10).log().mean()
        loss_c.backward()        
        optim_gen.step()    
        total_gen_loss += loss_c.data[0]
    graph()
    print(i, total_disc_loss /  X.shape[0], total_gen_loss / X.shape[0])