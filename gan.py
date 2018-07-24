from __future__ import print_function
import torch,torchvision,os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as Data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
#configures
dataroot="cifar10/"
if os.path.exists("cifar10"):
    down=False
else:
    down=True
if os.path.exists("imagenet")==False:
    os.makedirs("imagenet")
outf="."
imgSize=64
learning_rate=1e-3
beta1=0.9
batch_size=64
channel=3
workers=2
epoches=30
#####
g_channel=64
d_channel=64
z_size=100
#########
dataset=Data.CIFAR10(root=dataroot,download=down,transform=transforms.Compose((
    transforms.Scale(imgSize),transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ) ))
# dataroot="imagenet/"
# dataset=Data.ImageFolder(root=dataroot,transform=transforms.Compose((
#     transforms.Scale(imgSize),transforms.ToTensor(),transforms.CenterCrop(imgSize),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ) ))
assert dataset
dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers)

class GNet(nn.Module):
    def __init__(self):
        super(GNet,self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(z_size,g_channel*8,4,1,0,bias=False),
            nn.BatchNorm2d(8*g_channel),
            nn.ReLU(True),
            nn.ConvTranspose2d(8*g_channel,4*g_channel,4,2,1,bias=False),
            nn.BatchNorm2d(4*g_channel),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*g_channel,2*g_channel,4,2,1,bias=False),
            nn.BatchNorm2d(2*g_channel),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*g_channel,g_channel,4,2,1,bias=False),
            nn.BatchNorm2d(g_channel),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_channel,channel,4,2,1,bias=False),
            nn.Tanh()
        )
    def forward(self,x):
        return self.main(x)
    # def backward(self):
    #     pass

class DNet(nn.Module):
    def __init__(self):
        super(DNet,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(channel, d_channel, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_channel, 2 * d_channel, 4, 2, 1),
            nn.BatchNorm2d(2 * d_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * d_channel, 4 * d_channel, 4, 2, 1),
            nn.BatchNorm2d(4 * d_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * d_channel, 8 * d_channel, 4, 2, 1),
            nn.BatchNorm2d(8 * d_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * d_channel, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    def forward(self,x):
        return (self.main(x)).view(-1,1)
    # def backward(self):
    #     pass

################function to initialize the net
def init(l):
    name=l.__class__.__name__
    if name.find("Conv")!=-1:
        l.weight.data.normal_(0.0,0.02)
    elif name.find("BatchNorm")!=-1:
        l.weight.data.normal_(1.0,0.02)
        l.bias.data.fill_(0)
###############
G=GNet()
D=DNet()
G.apply(init)
D.apply(init)
print(G)
print(D)
criterion=nn.BCELoss()
####input
x=torch.FloatTensor(batch_size,3,imgSize,imgSize)
####noise
z=torch.FloatTensor(batch_size,z_size,1,1)
z_fix=torch.FloatTensor(batch_size,z_size,1,1).normal_(0,1)
label=torch.FloatTensor(batch_size)
real=1
fake=0
D.cuda()
G.cuda()
criterion.cuda()
x,label=x.cuda(),label.cuda()
z,z_fix=z.cuda(),z_fix.cuda()

z_fix=Variable(z_fix)
d_optim=optim.Adam(D.parameters(),betas=(beta1,0.999),lr=learning_rate)
g_optim=optim.Adam(G.parameters(),betas=(beta1,0.999),lr=learning_rate)

for epoch in range(epoches):
    for i,data in enumerate(dataloader,0):
        d_optim.zero_grad()
        real1,_=data
        batch_size=real1.size(0)
        real1=real1.cuda()
        x.resize_as_(real1).copy_(real1)
        label.resize_(batch_size).fill_(real)
        xv=Variable(x)
        labelv=Variable(label)
        output=D(xv)
        eriri_D_real=criterion(output,labelv)
        eriri_D_real.backward()
        D_x=output.data.mean()

        z.resize_(batch_size,z_size,1,1).normal_(0,1)
        zv=Variable(z)
        labelv=Variable(label.fill_(fake))
        fake1=G(zv)
        output=D(fake1.detach())
        eriri_D_fake=criterion(output,labelv)
        eriri_D_fake.backward()
        D_G_z1=output.data.mean()
        eriri_D=eriri_D_fake+eriri_D_real
        d_optim.step()

        g_optim.zero_grad()
        labelv=Variable(label.fill_(real))
        output=D(fake1)
        eriri_G=criterion(output,labelv)
        eriri_G.backward()
        D_G_z2=output.data.mean()
        g_optim.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epoches, i, len(dataloader),
                 eriri_D.data[0], eriri_G.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real1,
                              '%s/real_samples.png' % outf,
                              normalize=True)
            fake1 = G(z_fix)
            vutils.save_image(fake1.data,
                              '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                              normalize=True)
    torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
