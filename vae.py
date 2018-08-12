import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from CustomMNIST import CustomMNIST
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import save_image

#Load in the training and testing images
mnist_trainset = CustomMNIST(root='./data/original_mnist',train=True,process=True,transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset = CustomMNIST(root='./data/original_mnist',train=False,process=True,transform=transforms.Compose([transforms.ToTensor()]))

#Initialise the data loaders for training and testing
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=100, shuffle=True)
test_loader  = torch.utils.data.DataLoader(mnist_testset,batch_size=100, shuffle=True)

def flatten_tensor(x):
    length = 1
    for s in x.size()[1:]:
        length *= s
    return x.view(-1,length)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.training = True
        self.fc1  = nn.Linear(784,400)
        self.mean = nn.Linear(400,20)
        self.sd   = nn.Linear(400,20)
        self.fc2  = nn.Linear(20,400)
        self.fc3  = nn.Linear(400,784)

    def encode(self, x):
        y = F.relu(self.fc1(x))
        return self.mean(y), self.sd(y)

    def decode(self,z):
        return F.sigmoid(self.fc3(F.relu(self.fc2(z))))

    def reparameterise(self, mean, sd):
        #distr = Normal(0,1)
        if self.training:
            std = torch.exp(0.5*sd)
            e = torch.randn_like(sd)
            return e.mul(std).add_(mean)

        return mean

    def forward(self,x):
        x = flatten_tensor(x)
        mean, sd = self.encode(x)
        z = self.reparameterise(mean,sd)
        return self.decode(z), mean , sd




def calculate_loss(original, reconstructed, mean, log_var):
    recon_loss = F.binary_cross_entropy(reconstructed, original.view(-1,784), size_average=False)
    #KLD = -0.5 * torch.sum( 1 - mean.pow(2) + log_var - log_var.exp())
    return recon_loss
    #return recon_loss + KLD

model = VAE()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

def train(epoch):
    model.train()
    model.training = True
    running_loss = 0.0
    for i , (original, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon, mean, sd = model(original)
        loss = calculate_loss(original,recon,mean,sd)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(running_loss/len(train_loader))

def test(epoch):
    model.training = False
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (original,_) in enumerate(test_loader):
            recon, mean, sd = model(original)
            #print(recon.size())
            #print(original.size())
            loss = calculate_loss(original,recon,mean,sd)
            running_loss += loss.item()
            if i == len(test_loader) - 1:
                n = min(original.size(0),8)
                compare = torch.cat([original[:n], recon.view(100,1,28,28)[:n]])
                save_image(compare.cpu(),'Graphs/vaeMod/reconstruction' + str(epoch) + '.png', nrow=n)

        print(running_loss/len(test_loader))

for i in range(1,10):
    print(i)
    train(i)
    test(i)
    if (i % 50 == 0):
        with torch.no_grad():
            sample = torch.rand(64,30)
            sample = model.decode(sample)
            save_image(sample.view(64,1,28,28), 'Graphs/vaeMod/sample ' + str(i) + '.png')




