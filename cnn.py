import struct
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from CustomMNIST import CustomMNIST
from torch.autograd import Variable
import pandas as pd
import numpy as np


#Download the mnist data used for training and testing the network

#mnist_trainset = datasets.MNIST(root='./data/original_mnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#mnist_testset  = datasets.MNIST(root='./data/original_mnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_trainset = CustomMNIST(root='./data/modified_mnist',train=True,process=True,transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset = CustomMNIST(root='./data/modified_mnist',train=False,process=True,transform=transforms.Compose([transforms.ToTensor()]))

#Initialise the data loaders for training and testing
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=4, shuffle=True)
test_loader  = torch.utils.data.DataLoader(mnist_testset,batch_size=4, shuffle=True)

fig, ax = plt.subplots()
xData, yData = [] ,[]
ln, = plt.plot([],[], animated=True)


# Create MNIST datasets

def init_datasets(root):
    mnist_trainset = CustomMNIST(root=root, train=True, process=True, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_testset = CustomMNIST(root=root, train=False, process=True, transform=transforms.Compose([transforms.ToTensor()]))


#Resize tensor to appropriate shape to be compatible with fc layer

def flatten_tensor(x):
    length = 1
    for s in x.size()[1:]:
        length *= s
    return x.view(-1,length)

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet,self).__init__()
        self.pool    = nn.MaxPool2d(2,2)
        self.conv1  = nn.Conv2d(1,10,5)
        self.conv2  = nn.Conv2d(10,20,5)
        self.fc1    = nn.Linear(20* 4 * 4, 50)
        self.fc2    = nn.Linear(50,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten_tensor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class ResNet(nn.Module):

    def __init__(self, block, stacks,  num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1,64,5,stride=1,padding=(2,2))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2,2))
        self.bn = nn.BatchNorm2d(64)
        self.comp1 = self.create_component(block, 64, stacks[0])
        self.comp2 = self.create_component(block, 128, stacks[1], 2)
        self.comp3 = self.create_component(block, 256, stacks[2], 2)
        self.comp4 = self.create_component(block, 512, stacks[3], 2 )
        self.avgpool = nn.AvgPool2d((2,2), stride=1)
        self.fc = nn.Linear(512,num_classes)


    def create_component(self,block,out_planes,num_blocks,stride=1):
        layers = list()

        downSample = None
        if stride != 1:
            downSample = nn.Sequential(
                nn.Conv2d(self.in_planes,out_planes,kernel_size=1,stride=2),
                nn.BatchNorm2d(out_planes)
            )

        layers.append(block(self.in_planes,out_planes,stride,downSample))
        self.in_planes = out_planes
        for i in range(1,num_blocks):
            layers.append(block(self.in_planes,out_planes))

        return nn.Sequential(*layers)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.comp1(x)
        x = self.comp2(x)
        x = self.comp3(x)
        x = self.comp4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class Block(nn.Module):

    def __init__(self,inplanes,outplanes,stride=1,downSample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,outplanes,5,stride=stride,padding=2)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outplanes,outplanes,5,stride=1,padding=2)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downSample = downSample
        self.stride = stride
        self.inplanes = inplanes
        self.outplanes =outplanes

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downSample is not None:
            residual = self.downSample(residual)

        out += residual
        out = self.relu(out)

        return out


class VGG(nn.Module):
    
    def __init__(self):
        super(VGG, self).__init__()
        self.pool   = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.adpool = nn.AdaptiveMaxPool2d((4,4))
        self.conv1  = nn.Conv2d(1,64,3,padding=(1,1))
        self.conv2  = nn.Conv2d(64,64,3,padding=(1,1))
        self.conv3  = nn.Conv2d(64,128,3,padding=(1,1))
        self.conv4  = nn.Conv2d(128,128,3,padding=(1,1))
        self.conv5  = nn.Conv2d(128,256,3,padding=(1,1))
        self.conv6  = nn.Conv2d(256,256,3,padding=(1,1))
        self.conv7  = nn.Conv2d(256,256,3,padding=(1,1))
        self.conv8  = nn.Conv2d(256,512,3,padding=(1,1))
        self.conv9  = nn.Conv2d(512,512,3,padding=(1,1))
        self.conv10  = nn.Conv2d(512,512,3,padding=(1,1))
        self.conv11  = nn.Conv2d(512,512,3,padding=(1,1))
        self.conv12  = nn.Conv2d(512,512,3,padding=(1,1))
        self.conv13  = nn.Conv2d(512,512,3,padding=(1,1))
        self.fc1    = nn.Linear(512, 4096)
        self.fc2    = nn.Linear(4096,4096)
        self.fc3    = nn.Linear(4096,10)


    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(nn.BatchNorm2d(64)(x))
        x = self.conv2(x)
        x = F.relu(nn.BatchNorm2d(64)(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(nn.BatchNorm2d(128)(x))
        x = self.conv4(x)
        x = F.relu(nn.BatchNorm2d(128)(x))
        x = self.pool(x)
        x = self.conv5(x)
        x = F.relu(nn.BatchNorm2d(256)(x))
        x = self.conv6(x)
        x = F.relu(nn.BatchNorm2d(256)(x))
        x = self.conv7(x)
        x = F.relu(nn.BatchNorm2d(256)(x))
        padding = Variable(torch.zeros(4,256,1,7))
        x = torch.cat((x,padding),2)
        padding = Variable(torch.zeros(4,256,8,1))
        x = torch.cat((x,padding),3)
        x = self.adpool(x)
        x = self.conv8(x)
        x = F.relu(nn.BatchNorm2d(512)(x))
        x = self.conv9(x)
        x = F.relu(nn.BatchNorm2d(512)(x))
        x = self.conv10(x)
        x = F.relu(nn.BatchNorm2d(512)(x))
        x = self.pool(x)
        x = self.conv11(x)
        x = F.relu(nn.BatchNorm2d(512)(x))
        x = self.conv12(x)
        x = F.relu(nn.BatchNorm2d(512)(x))
        x = self.conv13(x)
        x = F.relu(nn.BatchNorm2d(512)(x))
        x = self.pool(x)

        x = flatten_tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim=0)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784,600)
        self.fc2 = nn.Linear(600,300)
        self.fc3 = nn.Linear(300,150)
        self.fc4 = nn.Linear(150,10)

    def forward(self, x):
        x = flatten_tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(x, dim = 0)
        return x




def train(model,optimizer,criterion,x,y):
    #x = []
    #y = []
    #for e in range(epoch):
    #print(e)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        labels = labels.long()
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    #print(running_loss/len(train_loader))
    x.append(len(x) + 1)
    y.append(running_loss/len(train_loader))


    #plot_loss(x,y)

    #print('Finished training')
    #return [x,y]



# Runs the training loop for 'epoch' number of times
def add_loss_point(frame, *fargs):

    model = fargs[0]
    optimizer = fargs[1]
    criterion = fargs[2]
    running_loss = 0.0
    x = train_loader
    y = 0

    #for i, data in enumerate(train_loader, 0):

    data = fargs[5][frame][1]
    inputs, labels = data
    optimizer.zero_grad()

    outputs = model(inputs)
    labels = labels.long()
    loss = criterion(outputs, labels)
    running_loss += loss.item()
    loss.backward()
    optimizer.step()

    print(running_loss)
    print(y)
    print('here')

    fargs[3].append(frame)
    fargs[4].append(running_loss)
    ln.set_data(fargs[3], fargs[4])

    return ln,


# Compares the output from the network against the test data to assess accuracy
def test(model,criterion,x,y):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    running_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            labels = labels.long()
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()

            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        x.append(len(x) + 1)
        y.append(running_loss/len(test_loader))
'''
    accuarcies =[]
    for i in range(10):
        accuarcies.append(100 * class_correct[i] /class_total [i])
        #print('Accuracy of class %d was: %d %%' %(i,(100 * class_correct[i] / class_total[i])))
    #print('Accuracy of network on the 10000 test images: %d %%' %(100 * correct / total))
    return accuarcies

'''

def plot_loss(xVals, yVals):

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(xVals, yVals)



def init():
    ax.set_xlim(0,2500)
    ax.set_ylim(0,3)
    return ln,


#optimizer = optim.Adamax(model.parameters(),0.0001)
#optimizer = optim.Adadelta(model.parameters(), 0.0001)
#optimizer = optim.SGD(model.parameters(),0.0001,0.8)
#optimizer = create_optimizer(0.0001, model, 0.8)
#train(model,1,optimizer,criterion)



def show_loss():
    args = (model, optimizer, criterion, xData, yData, list(enumerate(train_loader)))
    ani = FuncAnimation(fig, add_loss_point, frames=range(2500), fargs=args, init_func=init, blit=True)
    plt.show()


def run_model(model,epoch,name):
    test_loss_x = []
    test_loss_y = []
    train_loss_x = []
    train_loss_y = []
    opt = optim.SGD(model.parameters(), 0.0001,0.8)
    criterion = nn.CrossEntropyLoss()
    for e in range(epoch):
        print(e)
        train(model,opt,criterion,train_loss_x,train_loss_y)
        test(model,criterion,test_loss_x,test_loss_y)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(train_loss_x,train_loss_y)
    plt.savefig('Graphs/discriminator/' + name + 'train_loss.png')
    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(test_loss_x,test_loss_y)
    plt.savefig('Graphs/discriminator/' + name + '_test_loss.png')
    plt.clf()

'''

x = range(0,100)
y = x
print(x)
print(y)
plt.plot(x,y)
plt.savefig('Graphs/linear.png')
plt.clf()
y = np.square(x)
plt.plot(x,y)
plt.savefig('Graphs/quadratic.png')
'''

#model = LeNet()
#run_model(model,40,'lenet')
model = MLP()
run_model(model,20,'mlp')
#model = ResNet(Block,[2,2,2,2],10)
#run_model(model,40,'resnet')

    




'''




#model = ResNet(Block,[2,2,2,2],10)
losses = []
accuracies = []
criterion = nn.CrossEntropyLoss()

lr = 0.001
momentum = 0.8
#run_model(LeNet(), losses, accuracies, lr, momentum, criterion)
#run_model(MLP(), losses, accuracies, lr, momentum, criterion)
#run_model(ResNet(Block,[2,2,2,2],10), losses, accuracies, lr, momentum, criterion)
#run_model(LeNet(), losses, accuracies, lr, momentum, criterion)


plt.xlabel('epoch')
plt.ylabel('loss')


plt.plot(losses[0][0], losses[0][1])
plt.plot(losses[1][0], losses[1][1])
plt.plot(losses[2][0], losses[2][1])
plt.plot(losses[3][0], losses[3][1])
plt.legend(['LeNet','MLP','ResNet','VGG'],loc='upper right')
plt.savefig('Graphs/originalLoss.png')
#plt.show()



x = ['0','1','2','3','4','5','6','7','8','9']


df = pd.DataFrame(np.c_[accuracies[0],accuracies[1]], accuracies[2], accuracies[3], index=x)
print(df)
ax = df.plot.bar()
ax.legend(['LeNet','MLP','ResNet','VGG'])
plt.savefig('Graphs/originalAccuracy.png')
#plt.show()


'''
