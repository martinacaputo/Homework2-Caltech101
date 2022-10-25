from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        self.data=[]
        
        with open('train.txt') as f:
            for line in f:
                lab=line.split('/')[0]
                img=line.split('/')[1].strip()
                image=pil_loader(root+'/'+lab+'/'+img)
                self.data.append((image,lab))
        
        


     
        return self

    def __getitem__(self, index):

        image, label = self.data[index] # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
