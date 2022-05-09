import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from conv_vae import ConVAE#, VAE_loss_function
import torchvision.transforms.functional as TF

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

"""
Initialize Hyperparameters
"""
batch_size = 16
# learning_rate = 1e-4
learning_rate = 5e-6
num_epochs = 20
alpha = 0.01 # noise level

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

"""
Dataloader
"""

class ImgPrep:
    def __init__(self):
        pass
    def __call__(self, x):
        # x = TF.to_grayscale(x)
        # x = TF.adjust_sharpness(x, 10)
        x = TF.adjust_contrast(x, 2)
        x = TF.autocontrast(x)
        # x = TF.gaussian_blur(x, 3)
        x = TF.equalize(x)
        # x = TF.posterize(x, 1)
        return x

# data_path = 'archive/coins/data/train'
# data_path = 'RRC-60/Observe/'
# data_path = 'lfw-deepfunneled'
data_path = 'data/'
transform = transforms.Compose([transforms.Resize(128),
                                transforms.CenterCrop(128),
                                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                ImgPrep(),
                                transforms.ToTensor(),
                                ])
dataset = datasets.ImageFolder(data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



# """
# Create dataloaders to feed data into the neural network
# Default MNIST dataset is used and standard train/test split is performed
# """
# # train_loader = torch.utils.data.DataLoader(
# #     datasets.MNIST('data', train=True, download=True,
# #                     transform=transforms.ToTensor()),
# #     batch_size=batch_size, shuffle=True)
# # test_loader = torch.utils.data.DataLoader(
# #     datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
# #     batch_size=1)


"""
Initialize the network and the Adam optimizer
"""
model = ConVAE().to(device)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
model.train()
for epoch in range(num_epochs):
    # print(len(train_loader))
    for idx, data in enumerate(train_loader):
        # print(idx)
        imgs, labels = data
        # add noise
        imgs_wn = imgs + alpha * torch.randn(imgs.size())
        # print(labels.size())
        imgs = imgs.to(device)
        imgs_wn = imgs_wn.to(device)

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = model(imgs_wn)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.mean(-1 - logVar + mu.pow(2) + logVar.exp())
        # loss = 20 * F.binary_cross_entropy(out, imgs) + kl_divergence
        loss = 10 * F.mse_loss(out, imgs) + kl_divergence
        # loss = VAE_loss_function(out, imgs, mu, logVar)
        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))