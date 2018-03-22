import torch
from torch.autograd import Variable as V
import argparse
from util import load_mnist
from torch.distributions import Normal
from tqdm import tqdm
from cgan_model import DC_Discriminator, DC_Generator
import numpy as np
import torchvision.utils as vutils

np.random.seed(1111)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--latdim", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--beta1", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--bsize", type=int, default=128)
    return parser.parse_args()

args = parse_args()

def train_gan(G, D, train_loader, optim_disc, optim_gen, seed_dist):
    for t in tqdm(train_loader):
        img, y = t
        if img.size()[0] < args.bsize : continue
        if args.devid >= 0:
            img = img.cuda()
            y = y.cuda()
    
        # Grad real
        # -E[log(D(x))]
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        d = D(img, y)
        loss_a = -d.log().mean()
        loss_a.backward()
        
        # Grad fake
        # -E[log(1 - D(G(z)) )]
        seed = seed_dist.sample()
        if args.devid >= 0:
            seed = seed.cuda()
        
        x_fake = G(seed, y)
        d = D(x_fake.detach(), y)     
        loss_b = -(1 - d + 1e-10).log().mean()
        loss_b.backward()
        optim_disc.step()
       
        # Grad generator
        # Generator maximizes log probability of discriminator being mistaken
        # This trick deals with generator's vanishing gradients
        # See NIPS 2016 Tutorial: Generative Adversarial Networks 
        optim_disc.zero_grad()
        d = D(x_fake, y)
        loss_c = -d.log().mean()
        loss_c.backward()
        
        optim_gen.step()  

if __name__ == "__main__":
    train_loader = load_mnist(args.bsize)
    
    G = DC_Generator(1, args.ny, args.ngf, nz=args.latdim)
    D = DC_Discriminator(1, args.ndf, ny=args.ny, dim=784)
    optim_gen = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))    
    optim_disc = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    seed_distribution = Normal(V(torch.zeros(args.bsize, args.latdim)), 
                               V(torch.ones(args.bsize, args.latdim)))
    
    seed_distribution_test = Normal(V(torch.zeros(64, args.latdim)), 
                               V(torch.ones(64, args.latdim)))
    
    if args.devid >= 0:
        G.cuda()
        D.cuda()
    
    #Train
    print("Training..")
    for epoch in range(args.epoch):
        print("Training Epoch {}".format(epoch + 1))
        train_gan(G, D, train_loader, optim_disc, optim_gen, seed_distribution)
        
        seed = seed_distribution_test.sample()   
        y = torch.ones(64).long()
        fake = G(seed, y).unsqueeze(1).view(64, 1, 28, 28)
                
        vutils.save_image(fake.data,
            'cGAN_samples_local.png',
            normalize=True)
        
        
    torch.save(G.state_dict(), "G_local.pth")
    torch.save(D.state_dict(), "D.pth")