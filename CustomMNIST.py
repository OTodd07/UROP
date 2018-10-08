import os
import torch.utils.data as data
import torch
from PIL import Image
from mnist import MNIST
import numpy as np
import codecs


class CustomMNIST(data.Dataset):


    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    #Initialise the correct data sets
    def __init__(self,root,transform=None,process=True,train=True,patho=False):
        self.patho = patho
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        if process:
            self.process()

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))



    #Return the length of the data loader
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    #Get a certain item from the dataset
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(),mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, target


    # Initialise either modified or original MNIST data
    def process(self):

        if self.patho:
            train_labels = 'train-patho-idx1-ubyte'
            test_labels  = 't10k-patho-idx1-ubyte'
        else:
            train_labels = 'train-labels-idx1-ubyte'
            test_labels  = 't10k-labels-idx1-ubyte'

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, train_labels))

        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, test_labels))
        )


        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)



#Following two functions need to check for special numbers to ensure the MNIST files are valid

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)
