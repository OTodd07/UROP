import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
import torch.optim as optim
import matplotlib.pyplot as plt

#Download the mnist data used for training and testing the network

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),]))
mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()]))

#Initialise the data loaders for training and testing
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=20000, shuffle=True)
test_loader  = torch.utils.data.DataLoader(mnist_testset,batch_size=4, shuffle=True)

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()


# Set up the layers to be used in the network

    def create_network(self):
        self.pool   = nn.MaxPool2d(2,2)
        self.conv1  = nn.Conv2d(1,10,3)
        self.conv2  = nn.Conv2d(10,20,4)
        self.fc1    = nn.Linear(20* 6 * 6, 50)
        #self.fc2    = nn.Linear(50,10)
        self.fc2    = nn.Linear(1 * 32 * 32,10)
        self.num    = 1

    def forward(self,x):
        '''
        #print(x.size())
        # x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten_tensor(x)
        x = F.relu(self.fc1(x))
        '''
        x = self.flatten_tensor(x)
        x = self.fc2(x)
        return x

# Resize tensor to appropriate shape to be compatible with fc layer
    def flatten_tensor(self,x):
        length = 1
        for s in x.size()[1:]:
            length *= s
        return x.view(-1,length)


def create_optimizer(lr,cnn,momentum):
    return optim.SGD(cnn.parameters(),lr,momentum)


# Runs the training loop for 'epoch' number of times
def train(epoch,optimizer,criterion):
    x = []
    y = []
    for e in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        x.append(e)
        y.append(running_loss/len(train_loader))
    plot_loss(x,y) 

    print('Finished training')

# Compares the output from the network against the test data to assess accuracy
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

def plot_loss(xVals, yVals):
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(xVals, yVals)
    plt.savefig('Graphs/NetworkConfig%s' %(cnn.num))
    text_file = open('Graphs/NetworkConfig%s.txt' %(cnn.num) , 'w')
    text_file.write('%s\nBatch size: %s ' %(cnn, train_loader.batch_size))
    text_file.close()
    
    
    


cnn = CNN()
cnn.create_network()
criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
#criterion = nn.KLDivLoss()
print(train_loader.batch_size)
optimizer = create_optimizer(0.001,cnn,0.8)
train(10,optimizer,criterion)
test()
