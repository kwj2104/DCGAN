import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--latdim", type=int, default=50)

    parser.add_argument("--optim", choices=["Adadelta", "Adam", "SGD"], default="SGD")

    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--bsize", type=int, default=10)
    return parser.parse_args()

args = parse_args()


def load_mnist():
    
    train_dataset = datasets.MNIST(root='./data/',
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)
    test_dataset = datasets.MNIST(root='./data/',
                               train=False, 
                               transform=transforms.ToTensor())
    
    train_dataset[0][0]
    
    
    # Why not take x > .5 etc?
    # Treat the greyscale values as probabilities and sample to get binary data
    torch.manual_seed(3435)
    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])
    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])
    
    # Split training and val set
    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()
    train_img = train_img[:10000]
    train_label = train_label[:10000]
    
    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.bsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.bsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.bsize, shuffle=True)
    
#    for datum in train_loader:
#        img, label = datum
#        print(img)
#        print(label)
#        print(img.size(), label.size())
#        break

    return train_loader, val_loader, test_loader

# Compute the variational parameters for q
class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(2, dim)
        self.linear2 = nn.Linear(dim, args.latdim)
        self.linear3 = nn.Linear(dim, args.latdim)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        
        #Returns mean and variance (??)
        #Why is variance dim 8? Spherical gaussian so you can just input diagnal?
        return self.linear2(h), self.linear3(h)
    
    # Implement the generative model p(x | z)
class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(args.latdim, dim)
        self.linear2 = nn.Linear(dim, 2)

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
        
        # Where does the mul(.5) come from?
        q_normal = Normal(loc=mu, scale=logvar.mul(0.5).exp())
    

        # Reparameterized sample.
        z_sample = q_normal.rsample()
        #z_sample = mu
        return self.decoder(z_sample), q_normal

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_mnist()
    
    