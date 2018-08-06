import torch.nn as nn
import torch.optim as optim
from CustomMNIST import CustomMNIST
import torch
from torchvision.utils import save_image
from torchvision import transforms


mnist_trainset = CustomMNIST(root='./data/modified_mnist',train=True,process=True,transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=4, shuffle=True)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,100,4,2,1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(100,50,4,2,1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(50,10,4,2,2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(10,1,4,1,0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1,1).squeeze(1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(100,200,4,1,0),
            nn.ReLU(),
            nn.ConvTranspose2d(200,50,4,1,0),
            nn.ReLU(),
            nn.ConvTranspose2d(50,25,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(25,1,4,2,1),
            nn.Tanh()
        )



    def forward(self, x):
        return self.layers(x)

fixed_noise = torch.randn(4,100,1,1)
net_D = Discriminator()
net_G = Generator()
criterion = nn.BCELoss()
optimizer_D = optim.Adamax(net_D.parameters(),0.0001)
optimizer_G = optim.Adamax(net_G.parameters(),0.0001)
real_label = 1
fake_label = 0

for epoch in range(100):
    print(epoch)
    for i,data in enumerate(train_loader,0):
        net_D.zero_grad()
        real_img = data[0]
        batch_size = real_img.size(0)
        label = torch.full((batch_size,),real_label)

        #print(real_img.size())
        out = net_D(real_img)
        #print(out.size())
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
    save_image(fake.detach(), 'Graphs/gan/fake_sample' + str(epoch) + '.png')





