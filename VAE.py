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
import seaborn as sns; sns.set(color_codes=True)
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--latdim", type=int, default=20)
    parser.add_argument("--layersize", type=int, default=400)
    parser.add_argument("--alpha", type=float, default=1)

    parser.add_argument("--optim", choices=["Adadelta", "Adam", "SGD"], default="SGD")

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--bsize", type=int, default=64)
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
    
    
    
    # Why not take x > .5 etc?
    # Treat the greyscale values as probabilities and sample to get binary data
    torch.manual_seed(3435)
#    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
#    train_label = torch.LongTensor([d[1] for d in train_dataset])
#    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
#    test_label = torch.LongTensor([d[1] for d in test_dataset])
    
    train_img = torch.stack([d[0] for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])
    test_img = torch.stack([d[0] for d in test_dataset])
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
        self.linear1 = nn.Linear(dim, args.layersize)
        self.linear2 = nn.Linear(args.layersize, args.latdim)
        self.linear3 = nn.Linear(args.layersize, args.latdim)

    def forward(self, x):

        
        h = F.relu(self.linear1(x))
        
        #Returns mean and variance (??)
        #Why is variance dim 8? Spherical gaussian so you can just input diagnal?
        return self.linear2(h), self.linear3(h)
    
    # Implement the generative model p(x | z)
class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(args.latdim, args.layersize)
        self.linear2 = nn.Linear(args.layersize, dim)
        #self.sigmoid = nn.Sigmoid()

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
# Get samples.
p = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                        V(torch.ones(args.bsize, args.latdim)))

seed_distribution = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                        V(torch.ones(args.bsize, args.latdim)))

def train(train_loader, model, loss_func):
    model.train()

    total_loss = 0
    total_kl = 0
    total = 0
    alpha = args.alpha
    for t in train_loader:
        img, label = t
        batch_size = img.size()[0]
        if batch_size == args.bsize:
            print(img[0])
            raise Exception()
            
            # Standard setup. 
            model.zero_grad()
            x = img.view(args.bsize, -1)
    
            # Run VAE. 
            out, q = model(x)
            kl = kl_divergence(q, p).sum()
            rec_loss = loss_func(out, x)
            
            loss = rec_loss + alpha * kl 
            loss = loss / batch_size
    
            # record keeping.
            total_loss += loss_func(out, x).data / batch_size
            total_kl += kl.data / batch_size
            total += 1
            loss.backward()
            optim.step()
        
    return total_loss / total, total_kl / total

def val(val_loader, model, loss_func):
    model.eval()

    total_loss = 0
    total_kl = 0
    total = 0
    alpha = args.alpha
    for t in val_loader:
        img, label = t
        batch_size = img.size()[0]
        if batch_size == args.bsize:
        
            x = img.view(args.bsize, -1)
    
            # Run VAE. 
            out, q = model(x)
            kl = kl_divergence(q, p).sum()
            
            rec_loss = loss_func(out, x)
            
            loss = rec_loss + alpha * kl 
            loss = loss / args.bsize
    
            # record keeping.
            total_loss += loss_func(out, x).data / args.bsize
            total_kl += kl.data / args.bsize
            total += 1
            
            #look at test sample
            #if total % 10 == 0:
            #    fig, ax = plt.subplots()
            #    ax.matshow(out.view(args.bsize, 1, 28, 28).data[0][0], cmap=plt.cm.Blues)

            
    return total_loss / total, total_kl / total



if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_mnist()
    
    encoder = Encoder(dim=784)
    decoder = Decoder(dim=784)
    model = NormalVAE(encoder, decoder)
    
    loss_func = nn.BCEWithLogitsLoss(size_average=False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    
    #Train
    i = 0
    print("Training..")
    for epoch in range(args.epoch):
        rec_loss, kl_loss = train(train_loader, model, loss_func)
        #print("Train ==> Epoch: {} Reconstruction Loss: {} KL Loss: {}".format(i, rec_loss, kl_loss))
        rec_loss_val, kl_loss_val = val(val_loader, model, loss_func)
        print("Val ==> Epoch: {} Reconstruction Loss: {} KL Loss: {}".format(i, rec_loss_val, kl_loss_val))
        i += 1
        
    #Validate
    print("Testing..")
    rec_loss, kl_loss = val(val_loader, model, loss_func)
    print("Epoch: {} Reconstruction Loss: {} KL Loss: {}".format(i, rec_loss, kl_loss))
    
    #Save model
    
    
    #Sample some new pics
    seed = seed_distribution.sample()
    x = decoder(seed).view(args.bsize, 1, 28, 28)
    for i in range(30):
        fig, ax = plt.subplots()
        ax.matshow(x.data[i][0], cmap=plt.cm.Blues)
        
    
        
    
    
    
    
    
    