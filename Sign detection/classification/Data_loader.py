import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image

def load_traffic_sign_data(training_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    return X_train, y_train


train_tf = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor()
                               ])



class MyDataset(Dataset):
    def __init__(self, images=None, labels=None, transform=train_tf):
        super(MyDataset, self).__init__()

        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):

        image = self.images[index]

        label = self.labels[index]
        # image = image / 255
        # print(image.shape)
        image = Image.fromarray(image)
        # NORMALIZE 0 t0 1
        if self.transform is not None:
            image = self.transform(image)
            label = torch.from_numpy(np.array(int(label)))

        return image, label

    def __len__(self):
        return len(self.images)



# a,b = load_traffic_sign_data('E:\ex_python\sign_detection\\backup\\train.p')
#
# trainloader = DataLoader(
#     dataset=MyDataset(images=a,labels=b),
#     batch_size=4,
#     shuffle=True
#     )
#
# for index,(image,label) in enumerate(trainloader):
#     print(image.shape,label)