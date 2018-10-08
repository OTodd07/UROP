import os
import torch
from skimage import io, transform
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms,utils
import numpy as np




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



train_set = SkinDataset('data/Segmentation/ISIC2018_Task1-2_Training_Input')
train_loader = DataLoader(train_set,batch_size=4,shuffle=False)
ground_set = SkinDataset('data/Segmentation/ISIC2018_Task1_Training_GroundTruth')
ground_loader = DataLoader(train_set,batch_size=4,shuffle=False)

test = transforms.ToPILImage()
im = test(train_set[0][1])
im.show()
im = test(ground_set[0][1])
im.show()
