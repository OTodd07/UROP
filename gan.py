#!/usr/bin/env python

import torch.nn as nn
import torch.optim as optim
from CustomMNIST import CustomMNIST
import torch
from torchvision.utils import save_image
from torchvision import transforms


#mnist_trainset = CustomMNIST(root='./data/original_mnist',train=True,process=True,transform=transforms.Compose([transforms.ToTensor()]))
mnist_trainset = CustomMNIST(root='./data/original_mnist',train=True,process=True,transform=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=100, shuffle=True)
device = torch.device('cuda:0')
#print(torch.cuda.current_device().get_name())
#torch.backend.cudnn.enabled=False


def flatten_tensor(x):
    length = 1
    for s in x.size()[1:]:
        length *= s
    return x.view(-1,length)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layers = nn.Sequential(
            #nn.Linear(784,1024),
            #nn.LeakyReLU(0.1),
            #nn.Linear(1024,512),
            #nn.LeakyReLU(0.1),
            #nn.Dropout(0.3),
            #nn.Linear(512,256),
            #nn.LeakyReLU(0.1),
            #nn.Dropout(0.3),
            #nn.Linear(256,1),
            #nn.Dropout(0.3),
            #nn.Sigmoid()

            nn.Conv2d(1,128,4,2,1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,1024,4,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024,1,4,1,0),
            nn.Sigmoid()
        )

    def forward(self, x):
        #x = self.layers(x)
        output = nn.parallel.data_parallel(self.layers, x, range(1))
        return output.view(-1,1).squeeze(1)
        #return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(100,1024,4,1,0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(1024,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128,1,4,2,1),
            nn.Tanh()

        )

    '''
            nn.Linear(100,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,512),
            nn.LeakyReLU(0.1),
	        nn.Linear(512,1024),
	        nn.LeakyReLU(0.1),
            nn.Linear(1024,784),
            nn.Tanh()    
    '''

    '''
            nn.ConvTranspose2d(100,1024,4,1,0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(1024,512,4,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128,1,4,2,1),
            nn.Tanh()
    '''

    def forward(self, x):
        output =  nn.parallel.data_parallel(self.layers, x, range(1))
        return output
        #return self.layers(x)

fixed_noise = torch.randn(100,100,1,1, device=device)
net_D = Discriminator().to(device)
net_G = Generator().to(device)
criterion = nn.BCELoss()
optimizer_D = optim.Adam(net_D.parameters(), lr=0.0002)
optimizer_G = optim.Adam(net_G.parameters(), lr=0.0002)
#optimizer_D = optim.SGD(net_D.parameters(), lr=0.001, momentum=0.9)
#optimizer_G = optim.SGD(net_G.parameters(), lr=0.001, momentum=0.9)
real_label = 1
fake_label = 0

for epoch in range(100):
    print(epoch)
    for i,data in enumerate(train_loader,0):
        net_D.zero_grad()
        real_img = data[0].to(device)
        #print(real_img.size())
        batch_size = real_img.size(0)
        #print(batch_size)
        label = torch.full((batch_size,),real_label,device=device)

        #print(real_img.size())
        out = net_D(real_img)
        #print('heeeeeeeeeeeere')
        #print(out.size())
        #print(label.size())
        D_real_Err = criterion(out,label)
        D_real_Err.backward()


        noise = torch.randn(batch_size,100,1,1,device=device)
        #print(noise.size())
        fake = net_G(noise)
        #print(fake.size())
        label.fill_(fake_label)
        #print(fake.size())
        out = net_D(fake.detach())
        D_fake_Err = criterion(out,label)
        D_fake_Err.backward()
        optimizer_D.step()




        net_G.zero_grad()
        label.fill_(real_label)
        out = net_D(fake)
        G_Err = criterion(out,label)
        G_Err.backward()
        optimizer_G.step()

    fake = net_G(fixed_noise)
    #fake = fake.view(4,1,28,28)
    fake = fake.view(100,1,64,64)
    print(fake.size())
    save_image(fake.detach(), 'Graphs/gan/fake_sample' + str(epoch) + '.png')





