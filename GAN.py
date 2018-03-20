import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import argparse
from util import load_mnist
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--latdim", type=int, default=20)

    parser.add_argument("--optim", choices=["Adadelta", "Adam", "SGD"], default="SGD")

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--bsize", type=int, default=64)
    return parser.parse_args()

args = parse_args()

class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(args.latdim, 100)
        self.linear2 = nn.Linear(100, dim)
    def forward(self, z):
        return self.linear2(F.relu(self.linear1(z)))
    
class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(dim, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, point):
        return F.sigmoid(self.linear2(F.relu(self.linear1(point))))

def train_gan(G, D, train_loader, optim_disc, optim_gen, seed_dist):
    G.train()
    D.train()
    
    total_gen_loss = 0
    total_disc_loss = 0
    for t in train_loader:
        img, label = t
        if img.size()[0] < args.bsize : continue

        # Grad real
        # -E[log(D(x))]
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        d = D(img.view(args.bsize, -1))
        loss_a = 0.5 * -d.log().mean()
        loss_a.backward()
        
        # Grad fake
        # -E[log(1 - D(G(z)) )]
        seed = seed_dist.sample()
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
    
    return total_disc_loss /  img.size()[0], total_gen_loss / img.size()[0]

def val_gan(G, D, val_loader, optim_disc, optim_gen, seed_dist):
    G.eval()
    D.eval()
    
    total_gen_loss = 0
    total_disc_loss = 0
    for t in val_loader:
        img, label = t
        if img.size()[0] < args.bsize : continue

        # Grad real
        # -E[log(D(x))]
        d = D(img.view(args.bsize, -1))
        loss_a = 0.5 * -d.log().mean()
        
        # Grad fake
        # -E[log(1 - D(G(z)) )]
        seed = seed_dist.sample()
        x_fake = G(seed)
        d = D(x_fake.detach())        
        loss_b = 0.5 * -(1 - d + 1e-10).log().mean()
        total_disc_loss += loss_a.data[0] + loss_b.data[0]

        # Grad generator
        d = D(x_fake)
        loss_c = (1 - d + 1e-10).log().mean()  
        total_gen_loss += loss_c.data[0]
    
    return total_disc_loss /  img.size()[0], total_gen_loss / img.size()[0]

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_mnist(args.bsize)
    
    G = Generator(dim=784)
    D = Discriminator(dim=784)
    optim_gen = torch.optim.Adam(G.parameters(), lr=args.lr)
    optim_disc = torch.optim.Adam(D.parameters(), lr=args.lr)
    seed_distribution = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                               V(torch.ones(args.bsize, args.latdim)))
    
    #Train
    print("Training..")
    for epoch in tqdm(range(args.epoch)):
        #print("Epoch: {}".format(epoch))
        disc_loss, gen_loss = train_gan(G, D, train_loader, optim_disc, optim_gen, seed_distribution)
        #print("Train ==> Discriminator Loss: {} Generator Loss: {}".format(disc_loss, gen_loss))
        #disc_loss_val, gen_loss_val = val_gan(G, D, val_loader, optim_disc, optim_gen, seed_distribution)
        #print("Val ==> Discriminator Loss: {} Generator Loss: {}".format(disc_loss_val, gen_loss_val))
        #print()
    
    #Sample some new pics
    seed = seed_distribution.sample()
    x = G(seed).view(args.bsize, 1, 28, 28)
    for i in range(30):
        fig, ax = plt.subplots()
        ax.matshow(x.data[i][0], cmap=plt.cm.Blues)