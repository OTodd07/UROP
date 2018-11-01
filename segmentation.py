import os
import torch
from skimage import io, transform
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms,utils
import numpy as np
import torch.nn as nn
import torch.optim as optim




class SkinDataset(Dataset):

    def __init__(self, root_dir):
        self.root = root_dir
        self.x = 1
        self.transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])

    def __len__(self):
        path,dirs,files = next(os.walk(self.root))
        return len(files)


    def __getitem__(self, idx):
        count = 0
        img_name = ''
        files = sorted(os.listdir(self.root))
        print(files)
        img_name = os.path.join(self.root,files[idx + 1])

        img = Image.open(img_name)
        return img_name, self.transform(img)


class FCN(nn.Module):

    def __init__(self):
        self.layers = nn.Sequential(

            nn.Conv2d(1, 128, 4, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1, 4, 1, 0),


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
        )

    def forward(self, x):
        output = nn.parallel.data_parallel(self.layers, x , range(1))
        return output


def train():
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        labels = labels.to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    device = torch.device('cuda:0')
    train_set = SkinDataset('data/Segmentation/ISIC2018_Task1-2_Training_Input')
    train_loader = DataLoader(train_set,batch_size=4,shuffle=False)
    ground_set = SkinDataset('data/Segmentation/ISIC2018_Task1_Training_GroundTruth')
    ground_loader = DataLoader(train_set,batch_size=4,shuffle=False)
    model = FCN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0002)


test = transforms.ToPILImage()
im = test(train_set[0][1])
im.show()
im = test(ground_set[0][1])
im.show()
