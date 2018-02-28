from __future__ import print_function
import torch
import torch.utils.data
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

from DRAW import DRAW

# Load the MNIST dataset:
train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=2, shuffle=True)

model = DRAW()

def loss_function(c, m, s, T):

    LX = - c.sigmoid_().bernoulli().log()

    LZ = (m + s)/2.0 - T/2.0

    return LX+LZ

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):

    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)

        optimizer.zero_grad()
        c_T, mu_t, sigma_t = model(data)
        loss = loss_function(c_T, mu_t, sigma_t, model.T)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


for epoch in range(1, 100):
    train(epoch)

    sample = Variable(torch.randn(64, 20))

    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
