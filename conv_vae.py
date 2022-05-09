import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
"""
A Convolutional Variational Autoencoder
"""
class ConVAE(nn.Module):
    def __init__(self): # assume img is square.
        super(ConVAE, self).__init__()
        self.n1 = 64
        self.n2 = 64
        self.n3 = 32
        self.k = 5
        self.n_fc = 64
        self.n_channels = 3
        # self.n_channels = 1
        # self.fc_size = 128
        self.fc_size = 8192
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(self.n_channels, self.n1, self.k, padding=(2, 2))
        self.encConv2 = nn.Conv2d(self.n1, self.n2, self.k, padding=(2,2))
        self.encConv3 = nn.Conv2d(self.n2, self.n3, self.k, padding=(2,2))
        self.encFC1 = nn.Linear(self.fc_size, self.n_fc)
        self.encFC2 = nn.Linear(self.fc_size, self.n_fc)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(self.n_fc, self.fc_size)
        self.decConv3 = nn.ConvTranspose2d(self.n3, self.n2, self.k, padding=(2,2))
        self.decConv2 = nn.ConvTranspose2d(self.n2, self.n1, self.k, padding=(2,2))
        self.decConv1 = nn.ConvTranspose2d(self.n1, self.n_channels, self.k, padding=(2,2))

        self.pool = nn.MaxPool2d(2)
        self.upsampling = nn.Upsample(scale_factor=2)

        # self.bn1 = nn.BatchNorm2d(self.n1)
        # self.bn2 = nn.BatchNorm2d(self.n2)
        # self.bn3 = nn.BatchNorm2d(self.n3)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        # print(x.size())
        x = F.leaky_relu(self.encConv1(x))
        # x = self.bn1(x)
        x = self.pool(x)
        # print(x.size())
        x = F.leaky_relu(self.encConv2(x))
        # x = self.bn2(x)
        x = self.pool(x)
        # print(x.size())
        x = F.leaky_relu(self.encConv3(x))
        # x = self.bn3(x)
        x = self.pool(x)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        # print(x.size())
        x = x.view(-1, self.n3, 16, 16)
        # x = self.bn3(x)
        # print(x.size())
        x = self.upsampling(x)
        # print(x.size())
        x = F.leaky_relu(self.decConv3(x))
        # x = self.bn2(x)
        x = self.upsampling(x)
        # print(x.size())
        x = torch.sigmoid(self.decConv2(x))
        # x = self.bn1(x)
        x = self.upsampling(x)
        # print(x.size())
        x = torch.sigmoid(self.decConv1(x))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


# def VAE_loss_function(recon_x, x, mu, logvar):
#     # TO DO: Implement reconstruction + KL divergence losses summed over all elements and batch
#     mseloss = nn.MSELoss()
#     recon_loss = 30 * mseloss(recon_x, x)
#     KLD = - 0.5 * (1 + logvar - torch.square(mu) - torch.exp(logvar)).mean()
#     # see lecture 12 slides for more information on the VAE loss function
#     # for additional information on computing KL divergence
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114

#     return recon_loss + KLD