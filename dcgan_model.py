import torch.nn as nn

class DC_Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(DC_Discriminator,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=2),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=0, bias=True),
                                nn.Sigmoid())

    def forward(self,x):
        
        out = self.layer1(x)      
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out.squeeze(1).squeeze(1)
    
    
class DC_Generator(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(DC_Generator,self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz,ngf*4,kernel_size=3),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.ReLU())
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=3,stride=2,padding=0),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1),
                                 nn.Tanh())

    def forward(self,x):

        out = self.layer1(x)      
        out = self.layer2(out)   
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out