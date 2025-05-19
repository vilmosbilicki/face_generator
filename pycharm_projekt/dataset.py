import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import numpy as np
from matplotlib import pyplot as plt


class DataSet:
    def __init__(self, img_size, batch_size, path):
        self.img_size = img_size
        self.batch_size = batch_size
        self.path = path

        self.data_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Scales data into [0,1]
            transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
        ])
        self.reverse_transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2), # data back to [0,1]
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        self.train = torchvision.datasets.ImageFolder(root=path+"/train", transform=self.data_transform)
        self.test = torchvision.datasets.ImageFolder(root=path+"/test", transform=self.data_transform)

        self.data = torch.utils.data.ConcatDataset([self.train, self.test])
        self.dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=False, drop_last=True)



    def get_dataloader(self):
        return self.dataloader

    def show_tensor_image(self, image):
        #print(image.shape)
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        plt.imshow(self.reverse_transform(image))


