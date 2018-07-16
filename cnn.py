import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),]))
mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=10, shuffle=True)
test_loader  = torch.utils.data.DataLoader(mnist_testset,batch_size=4, shuffle=True)

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

    def create_network(self):
        self.pool   = nn.MaxPool2d(2,2)
        self.conv1  = nn.Conv2d(1,6,5)
        self.conv2  = nn.Conv2d(6,16,5)
        self.fc1    = nn.Linear(16* 5 * 5, 10)
        self.fc2    = nn.Linear(10,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten_tensor(x)
        x = self.flatten_tensor(x)
        x = self.flatten_tensor(x)
        print(x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def flatten_tensor(self,x):
        length = 1
        for s in x.size()[1:]:
            length *= s
        return x.view(1,length)

#params = list(cnn.parameters())


def create_optimizer(lr,cnn,momentum):
    return optim.SGD(cnn.parameters(),lr,momentum)

def train(epoch,optimizer,criterion):
    x = []
    y = []
    for e in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            print(inputs.size())
            print(labels.size())

            optimizer.zero_grad()

            outputs = cnn(inputs)
            #print(outputs.size())
            loss = criterion(outputs, labels)
            running_loss += loss.item()
           # print(loss.item())
            loss.backward()
            optimizer.step()
        #print(running_loss)
        x.append(e)
        y.append(running_loss/len(train_loader))
    print(x)
    print(y)
    plt.plot(x,y)

    print('Finished training')

def test():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of network on the 10000 test images: %d %%' %(100 * correct / total))



cnn = CNN()
cnn.create_network()
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
#criterion = nn.KLDivLoss()
optimizer = create_optimizer(0.001,cnn,0.8)
print(len(train_loader))
train(10,optimizer,criterion)
test()
plt.show()


