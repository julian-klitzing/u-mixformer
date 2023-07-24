from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data_utils

class DeepVIB(nn.Module):
    def __init__(self, input_shape, output_shape, z_dim):
        """
        Deep VIB Model.

        Arguments:
        ----------
        input_shape : `int`
            Flattened size of image. (Default=784)
        output_shape : `int`
            Number of classes. (Default=10)
        z_dim : `int`
            The dimension of the latent variable z. (Default=256)
        """
        super(DeepVIB, self).__init__()
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.z_dim  = z_dim

        # build encoder
        self.encoder = nn.Sequential(nn.Linear(input_shape, 1024),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True)                                    )
        self.fc_mu  = nn.Linear(1024, self.z_dim)
        self.fc_std = nn.Linear(1024, self.z_dim)

        # build decoder
        self.decoder = nn.Linear(self.z_dim, output_shape)

    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x):
        """
        Forward pass

        Parameters:
        -----------
        x : [batch_size,28,28]
        """
        # flattent image
        x_flat = x.view(x.size(0), -1)
        mu, std = self.fc_mu(x_flat), F.softplus(self.fc_std(x), beta=1)
        z = self.reparameterise(mu, std) # sample latent based on encoder outputs
        return self.decoder(z), mu, std

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix random seeds for reproducibility
seed = 73
torch.manual_seed(seed)
np.random.seed(seed)

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
# # 60000 tuples with 1x28x28 image and corresponding label
data = MNIST('data',
              train=True,
              download=True,
              transform = transforms.Compose([transforms.ToTensor()]))
# # Split data into images and labels
x_train = data.train_data
y_train = data.train_labels
# # Scale images from [0,255] to [0,+1]
x_train = x_train.float() / 255
# # Save as .npz
np.savez_compressed('data/mnist_train',
                     a=x_train,
                     b=y_train)

# # 10000 tuples with 1x28x28 image and corresponding label
data = MNIST('data',
              train=False,
              download=True,
              transform = transforms.Compose([transforms.ToTensor()]))
# # Split data into images and labels
x_test = data.test_data
y_test = data.test_labels
# # Scale images from [0,255] to [0,+1]
x_test = x_test.float() / 255
# # Save as .npz
np.savez_compressed('data/mnist_test',
                     a=x_test,
                     b=y_test)

# Load MNIST data locally
train_data = np.load('data/mnist_train.npz')
x_train = torch.Tensor(train_data['a'])
y_train = torch.Tensor(train_data['b'])
n_classes = len(np.unique(y_train))

test_data = np.load('data/mnist_test.npz')
x_test = torch.Tensor(test_data['a'])
y_test = torch.Tensor(test_data['b'])

# # Visualise data
# plt.rcParams.update({'font.size': 16})
# fig, axes = plt.subplots(1,4, figsize=(35,35))
# imx, imy = (28,28)
# labels   = [0,1,2,3]
# for i, ax in enumerate(axes):
#     visual = np.reshape(x_train[labels[i]], (imx,imy))
#     ax.set_title("Example Data Image, y="+str(int(y_train[labels[i]])))
#     ax.imshow(visual, vmin=0, vmax=1)
# plt.show()

##Training

# Hyperparameters
beta   = 1e-3
z_dim  = 256
epochs = 200
batch_size = 128
learning_rate = 1e-4
decay_rate = 0.97

# Create DatatLoader
train_dataset    = data_utils.TensorDataset(x_train, y_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

# Loss function: Cross Entropy Loss (CE) + beta*KL divergence
def loss_function(y_pred, y, mu, std):
    """
    y_pred : [batch_size,10]
    y : [batch_size,10]
    mu : [batch_size,z_dim]
    std: [batch_size,z_dim]
    """
    CE = F.cross_entropy(y_pred, y, reduction='sum')
    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
    print(f'CE: {CE} and KL: {KL}')
    return (beta*KL + CE) / y.size(0)

# Initialize Deep VIB
vib = DeepVIB(np.prod(x_train[0].shape), n_classes, z_dim)

# Optimiser
optimiser = torch.optim.Adam(vib.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=decay_rate)

# Send to GPU if available
vib.to(device)

print("Device: ", device)
print(vib)

# Training
measures = defaultdict(list)
start_time = time.time()

# put Deep VIB into train mode
vib.train()
for epoch in range(epochs):
    epoch_start_time = time.time()

    # exponential decay of learning rate every 2 epochs
    if epoch % 2 == 0 and epoch > 0:
        scheduler.step()

    batch_loss = 0
    batch_accuracy = 0
    for _, (X,y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.long().to(device)

        # Zero accumulated gradients
        vib.zero_grad()

        # forward pass through Deep VIB
        y_pred, mu, std = vib(X)

        # Calculate loss
        loss = loss_function(y_pred, y, mu, std) #mu and std로 KL divergence 계산
        # Backpropogation: calculating gradients
        loss.backward()
        # Update parameters of generator
        optimiser.step()

        # Save loss per batch
        batch_loss += loss.item()*X.size(0)
        # Save accuracy per batch
        y_pred = torch.argmax(y_pred,dim=1)
        batch_accuracy += int(torch.sum(y == y_pred))

    # Save losses per epoch
    measures['total_loss'].append(batch_loss / len(train_dataloader.dataset))
    # Save accuracy per epoch
    measures['accuracy'].append(batch_accuracy / len(train_dataloader.dataset))

    print("Epoch: {}/{}...".format(epoch+1, epochs),
          "Loss: {:.4f}...".format(measures['total_loss'][-1]),
          "Accuracy: {:.4f}...".format(measures['accuracy'][-1]),
          "Time Taken: {:,.4f} seconds".format(time.time()-epoch_start_time))

print("Total Time Taken: {:,.4f} seconds".format(time.time()-start_time))

#Test
# Create DatatLoader
test_dataset = data_utils.TensorDataset(x_test, y_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

measures = defaultdict(int)
start_time = time.time()

# put Deep VIB into train mode
vib.eval()
with torch.no_grad():
    for _, (X,y) in enumerate(test_dataloader):
        X = X.to(device)
        y = y.long().to(device)

        # forward pass through Deep VIB
        y_pred, _, _ = vib(X)

        y_pred = torch.argmax(y_pred,dim=1)
        measures['accuracy'] += int(torch.sum(y == y_pred))

print("Accuracy: {:.4f}...".format(measures['accuracy']/len(test_dataloader.dataset)),
      "Time Taken: {:,.4f} seconds".format(time.time()-start_time))