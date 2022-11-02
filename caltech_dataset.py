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
        super(Caltech, self).__init__(root+'/101_ObjectCategories', transform=transform, target_transform=target_transform)

        
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        self.data=[]
        lables=[]
        k=0
        j=0
        
        with open(root+'/train.txt') as f:
            for line in f:
                lab=line.split('/')[0]
                img=line.split('/')[1].strip()
                image=pil_loader(root+'/101_ObjectCategories'+'/'+lab+'/'+img)
                idl=0
                k=0
                if len(lables)==0:
                    lables.append((lab,j))
                    j=j+1
                for i in lables:
                    if i[0]==lab:
                        k=1
                        idl=i[1]
                if k==0:
                   lables.append((lab,j)) 
                   idl=j
                   j=j+1
                   
            
                self.data.append((image,idl))
        
        


     
        return None

    def __getitem__(self, index):

        (image, label) = self.data[index] 
        
                            # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return (image, label)

    def __len__(self):
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
