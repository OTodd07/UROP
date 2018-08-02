import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from CustomMNIST import CustomMNIST
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import save_image

#Load in the training and testing images
mnist_trainset = CustomMNIST(root='./data/modified_mnist',train=True,process=True,transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset = CustomMNIST(root='./data/modified_mnist',train=False,process=True,transform=transforms.Compose([transforms.ToTensor()]))

#Initialise the data loaders for training and testing
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=4, shuffle=True)
test_loader  = torch.utils.data.DataLoader(mnist_testset,batch_size=4, shuffle=True)

def flatten_tensor(x):
    length = 1
    for s in x.size()[1:]:
        length *= s
    return x.view(-1,length)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1  = nn.Linear(784,200)
        self.mean = nn.Linear(200,40)
        self.sd   = nn.Linear(200,40)
        self.fc2  = nn.Linear(40,200)
        self.fc3  = nn.Linear(200,784)

    def encode(self, x):
        y = F.relu(self.fc1(x))
        return self.mean(y), self.sd(y)

    def decode(self,z):
        return F.sigmoid(self.fc3(F.relu(self.fc2(z))))

    def reparameterise(self, mean, sd):
        distr = Normal(0,1)
        e = distr.sample()
        return mean + e * sd

    def forward(self,x):
        x = flatten_tensor(x)
        mean, sd = self.encode(x)
        z = self.reparameterise(mean,sd)
        return self.decode(z), mean , sd




def calculate_loss(original, reconstructed, mean, sd):
    recon_loss = F.binary_cross_entropy(reconstructed,original)
    KLD = -0.5 * torch.sum( 1 - mean.pow(2) + (sd.pow(2)).log() - sd.pow(2))
    return recon_loss + KLD


model = VAE()
optimizer = optim.Adamax(model.parameters(),0.0001)

def train(epoch):
    running_loss = 0.0
    for i , (original, _) in enumerate(train_loader):
        recon, mean, sd = model(original)
        loss = calculate_loss(original,recon,mean,sd)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

def test(epoch):
    running_loss = 0.0
    with torch.no_grad():
        for i, (original,_) in enumerate(test_loader):
            recon, mean, sd = model(original)
            loss = calculate_loss(original,recon,mean,sd)
            running_loss += loss.item()
            if i == 0:
                n = min(original.size(0), 4)
                compare = torch.cat([original[:n], recon.view(4,1,28,28)[:n]])
                save_image(compare.cpu(),'Graphs/reconstruction' + str(epoch) + '.png', nrow=n)


for i in range(1,10):
    print(i)
    train(i)
    test(i)
    with torch.no_grad():
        sample = torch.rand(30,20)
        sample = model.decode(sample)
        save_image(sample.view(30,1,28,28), 'Graphs/sample ' + str(i) + '.png')



