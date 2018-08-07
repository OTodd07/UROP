#!/usr/bin/env python

import torch.nn as nn
import torch.optim as optim
from CustomMNIST import CustomMNIST
import torch
from torchvision.utils import save_image
from torchvision import transforms


mnist_trainset = CustomMNIST(root='./data/original_mnist',train=True,process=True,transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=4, shuffle=True)


def flatten_tensor(x):
    length = 1
    for s in x.size()[1:]:
        length *= s
    return x.view(-1,length)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784,400),
            nn.LeakyReLU(0.1),
            nn.Linear(400,200),
            nn.LeakyReLU(0.1),
            nn.Linear(200,1),
            nn.Sigmoid()    
            #nn.Conv2d(1,100,4,2,1),
            #nn.LeakyReLU(0.1),
            #nn.Conv2d(100,50,4,2,1),
            #nn.LeakyReLU(0.1),
            #nn.Conv2d(50,10,4,2,2),
            #nn.LeakyReLU(0.1),
            #nn.Conv2d(10,1,4,1,0),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(flatten_tensor(x))
        return x.view(-1,1).squeeze(1)
        #return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layers = nn.Sequential(

            nn.Linear(100,200),
            nn.LeakyReLU(0.1),
            nn.Linear(200,400),
            nn.LeakyReLU(0.1),
            nn.Linear(400,784),
            nn.Tanh()    

        )


    '''
            nn.ConvTranspose2d(100,200,4,1,0),
            nn.ReLU(),
            nn.ConvTranspose2d(200,50,4,1,0),
            nn.ReLU(),
            nn.ConvTranspose2d(50,25,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(25,1,4,2,1),
            nn.Tanh()
    '''

    def forward(self, x):
        return self.layers(flatten_tensor(x))

fixed_noise = torch.randn(4,100,1,1)
net_D = Discriminator()
net_G = Generator()
criterion = nn.BCELoss()
optimizer_D = optim.Adam(net_D.parameters(), lr=1e-3)
optimizer_G = optim.Adam(net_G.parameters(), lr=1e-3)
real_label = 1
fake_label = 0

for epoch in range(100):
    print(epoch)
    for i,data in enumerate(train_loader,0):
        net_D.zero_grad()
        real_img = data[0]
        #print(real_img.size())
        batch_size = real_img.size(0)
        label = torch.full((batch_size,),real_label)

        #print(real_img.size())
        out = net_D(real_img)
        #print('heeeeeeeeeeeere')
        #print(out.size())
        #print(label.size())
        D_real_Err = criterion(out,label)
        D_real_Err.backward()


        noise = torch.randn(batch_size,100,1,1)
        #print(noise.size())
        fake = net_G(noise)
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
    fake = fake.view(4,1,28,28)
    print(fake.size())
    save_image(fake.detach(), 'Graphs/gan/fake_sample' + str(epoch) + '.png')





