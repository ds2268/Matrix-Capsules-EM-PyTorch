from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from random import shuffle
import os
from scipy.misc import imread, imsave, toimage, imresize
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image


def prepareDataset(dataset_path, ext="tiff"):
    train_data = []
    for subdir, dirs, files in os.walk(dataset_path):
        basename = os.path.basename(subdir)
        dirname = os.path.basename(os.path.dirname(subdir))
        for file in files:
            if file.endswith(ext):
                    train_data.append((os.path.join(subdir, file), basename.lower()))

    return train_data


def compute_normalization(train_data, size=(128, 128)):
    images = []
    for img_path in train_data:
        I = imread(img_path, mode="RGB")
        I = imresize(I, size)
        images.append(np.expand_dims(I / 255, 0))
    images = np.concatenate(tuple(images), 0)

    mean_image = np.mean(images, axis=0)
    std_image = np.std(images, axis=0)

    meanR = np.mean(images[:, :, :, 0].flatten())
    meanG = np.mean(images[:, :, :, 1].flatten())
    meanB = np.mean(images[:, :, :, 2].flatten())

    stdR = np.std(images[:, :, :, 0].flatten())
    stdG = np.std(images[:, :, :, 1].flatten())
    stdB = np.std(images[:, :, :, 2].flatten())

    return mean_image, std_image, (meanR, meanG, meanB), (stdR, stdG, stdB)


class ActionDataset(Dataset):

    def __init__(self, data, classes, transforms=None):
        self.transforms = transforms
        self.data = data
        self.classes = classes

    def __getitem__(self, index):
        img_path = self.data[index][0]
        img = Image.open(img_path).convert('L')

        if self.transforms is not None:
            img = self.transforms(img)

        label = self.classes[self.data[index][1]]

        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_data = prepareDataset("/Volumes/E_128/train")
    test_data = prepareDataset("/Volumes/E_128/test")
    print(len(train_data))
    print(len(test_data))

    classes = {
        "handwaving": 0,
        "handclapping": 1,
        "boxing": 2,
        "walking": 3,
        "running": 4,
        "jogging": 5
    }

    """mean_img, std_img, mean_c, std_c = compute_normalization(train_data)
    print(mean_img.shape)
    print(std_img.shape)
    print(mean_c)
    print(std_c)

    transformations = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                                          transforms.Normalize(mean_c, std_c)])"""
    action_dataset = ActionDataset(train_data, classes, transforms=transforms.Compose([transforms.ToTensor()]))

    action_dataset_loader = DataLoader(dataset=action_dataset,
                                       batch_size=1,
                                       shuffle=True)

    for images, labels in action_dataset_loader:
        print(images.size())

