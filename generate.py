import torch
from torch.distributions import Normal
import torchvision.utils as vutils
from torch.autograd import Variable as V
from DCGAN_model import DC_Generator
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--latdim", type=int, default=100)
    parser.add_argument('--G', default='G.pth', help="path to G (to continue training)")
    parser.add_argument("--bsize", type=int, default=64)
    return parser.parse_args()

args = parse_args()

def generate(bsize, latdim, decoder):
    
    seed_distribution = Normal(V(torch.zeros(bsize, latdim, 1, 1)), 
                               V(torch.ones(bsize, latdim, 1, 1)))
    
    seed = seed_distribution.sample()
    fake = decoder(seed)
    vutils.save_image(fake.data,
            'GAN_samples.png',
            normalize=True)
    
if __name__ == '__main__':
    
    assert args.G != '', "Generator Model must be provided!"
    
    G = DC_Generator(1, args.ngf, args.latdim)
    G.load_state_dict(torch.load("G.pth", map_location=lambda storage, loc: storage))
    generate(args.bsize, args.latdim, decoder=G)
    print("Generated sample saved to GAN_Samples.png")