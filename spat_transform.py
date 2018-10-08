import torch
from CustomMNIST import CustomMNIST
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Create MNIST datasets

def init_datasets(root,patho):
    mnist_trainset = CustomMNIST(root=root, train=True, process=True, patho=patho, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_testset = CustomMNIST(root=root, train=False, process=True, patho=patho, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset,batch_size=4, shuffle=True)
    return train_loader,test_loader



class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.pool    = nn.MaxPool2d(2,2)
        self.conv1  = nn.Conv2d(1,10,5)
        self.conv2  = nn.Conv2d(10,20,5)
        self.fc1    = nn.Linear(20* 4 * 4, 50)
        self.fc2    = nn.Linear(50,10)

        self.convlayers = nn.Sequential(
            nn.Conv2d(1,10,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(10,20,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.fc  = nn.Sequential(
            nn.Linear(20* 4 * 4, 50),
            nn.ReLU(),
            nn.Linear(50,10)
        )

        #Localisation network
        self.local = nn.Sequential (
            nn.Conv2d(1,10,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(10,20,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        #3x2 affine matrix
        self.regressor = nn.Sequential (
            nn.Linear(10 * 3 * 3, 32),
            nn.Tanh(),
            nn.Linear(32,3*2)
        )


        #Set up weights and bias
        self.regressor[2].weight.data.zero_();
        self.regressor[2].bias.data.copy_(torch.tensor[1,0,0,0,1,0])

        #Set up spatial transformer network forward pass
        def transformer(self,x):
            out = self.local(x)
            out = out.view(-1, 10 * 3 * 3)
            theta = self.local(out)
            theta = theta.view(-1,2,3)

            grid = F.affine_grid(theta,x.size)
            sample  = F.grid_sample(grid)
            return sample




