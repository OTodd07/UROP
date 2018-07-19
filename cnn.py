import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models

#Download the mnist data used for training and testing the network

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

#Initialise the data loaders for training and testing
train_loader = torch.utils.data.DataLoader(mnist_trainset,batch_size=4, shuffle=True)
test_loader  = torch.utils.data.DataLoader(mnist_testset,batch_size=4, shuffle=True)


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




class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()


# Set up the layers to be used in the network

    def create_Lenet(self):
        self.pool   = nn.MaxPool2d(2,2)
        self.conv1  = nn.Conv2d(1,10,5)
        self.conv2  = nn.Conv2d(10,20,5)
        self.fc1    = nn.Linear(20* 4 * 4, 50)
        self.fc2    = nn.Linear(50,10)
        self.num    = 3


    def create_VGG(self):
        self.pool   = nn.MaxPool2d(2,2)
        self.conv1  = nn.Conv2d(1,64,3,padding=(1,1))
        self.conv2  = nn.Conv2d(64,64,3,padding=(1,1))
        self.conv3  = nn.Conv2d(64,128,3,padding=(1,1))
        self.conv4  = nn.Conv2d(128,128,3,padding=(1,1))
        self.conv5  = nn.Conv2d(128,256,3,padding=(1,1))
        self.conv6  = nn.Conv2d(256,256,3,padding=(1,1))
        self.conv7  = nn.Conv2d(256,512,3,padding=(1,1))
        self.conv8  = nn.Conv2d(512,512,3,padding=(1,1))
        self.fc1    = nn.Linear(512, 4096)
        self.fc2    = nn.Linear(4096,4096)
        self.fc3    = nn.Linear(4096,10)
        self.num    = 3


    def forward_Lenet(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten_tensor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward_VGG(self,x):
        x = self.pool(F.relu(self.conv2(self.conv1(x))))
        x = self.pool(F.relu(self.conv4(self.conv3(x))))
        x = self.pool(F.relu(self.conv6(self.conv6(self.conv6(self.conv5(x))))))
        x = self.pool(F.relu(self.conv8(self.conv8(self.conv8(self.conv7(x))))))
        x = self.flatten_tensor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=0)
        return x


    def forward(self,x):
        return self.forward_VGG(x)

# Resize tensor to appropriate shape to be compatible with fc layer
    def flatten_tensor(self,x):
        length = 1
        for s in x.size()[1:]:
            length *= s
        return x.view(-1,length)


def create_optimizer(lr,cnn,momentum):
    return optim.SGD(cnn.parameters(),lr,momentum)


# Runs the training loop for 'epoch' number of times
def train(model,epoch,optimizer,criterion):
    x = []
    y = []
    for e in range(epoch):
        print(e)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs.cuda())
            loss = criterion(outputs.cuda(), labels.cuda())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        x.append(e)
        y.append(running_loss/len(train_loader))
    plot_loss(x,y) 

    print('Finished training')

# Compares the output from the network against the test data to assess accuracy
def test(model):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = model(images)
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
#model = models.alexnet(pretrained=False)
#model = models.resnet18(pretrained=False)
model = ResNet(Block, [2,2,2,2], 10)
model.cuda()
cnn.create_VGG()
criterion = nn.CrossEntropyLoss()
#criterion = F.nll_loss()
#criterion = nn.NLLLoss()
#criterion = nn.KLDivLoss()
print(torch.cuda.get_device_name(0))

print(train_loader.batch_size)
optimizer = create_optimizer(0.001,cnn,0.8)
train(model,2,optimizer,criterion)
test(model)

