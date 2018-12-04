#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6,16,5)
        # affine op: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        # Max pool in (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



net=Net()

input = torch.randn(1,1,32,32)
out = net(input)

target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(out, target)
net.zero_grad()

loss.backward()

learning_rate = 0.01

for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


print(list(net.parameters())[0])
