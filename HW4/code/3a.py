#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def train(net, trainloader, testloader, calc_acc, criterion, optimizer, rate,
          momentum, max_epochs=10,acc_file=None):
    splits = np.zeros(max_epochs+1)
    splits[0] = time.time()

    for epoch in range(max_epochs):
     
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
    
            optimizer.zero_grad()
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
            if i % 2000 == 1999:
                print(f'[{epoch+1:d}, {i+1:5d}] loss: {running_loss/2000:.3f}')
    
                running_loss = 0.
        
        splits[epoch+1] = time.time()
        print(f"Finished epoch {epoch} in {splits[epoch+1]-splits[epoch]:.0f}")
        print(f"Total time: {splits[epoch+1]-splits[0]:.0f}")

        if calc_acc:
            train_acc = accuracy(trainloader,net)
            test_acc = accuracy(testloader,net)

            with open(acc_file,'a') as f:
                f.write(f"{train_acc:10.4f}{test_acc:10.4f}\n")
        
            print(f"After epoch {epoch+1}:")
            print(f"Training accuracy: {train_acc:.2f}")
            print(f"Test accuracy: {test_acc:.2f}")
    
    print("Done training!")

def accuracy(loader,net):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100*correct/total

###############################################################################

class Net(nn.Module):
    def  __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(32*32*3,10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

###############################################################################
# arg processing and filename init                                            #                    
###############################################################################

n_epochs = int(sys.argv[1])
n_workers = int(sys.argv[2])
rate = float(sys.argv[3])
momentum = float(sys.argv[4])
calc_acc = {'y':True, 'n':False}[sys.argv[5]]

total_file = f"../data/3a_parameter_search"

if calc_acc:
    acc_file = f"../data/3a_r{100000*rate:.0f}_p{100000*momentum:.0f}_accs"
else:
    acc_file = None

###############################################################################
# load and normalize data                                                     #
############################################################################### 

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=n_workers)


testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True, num_workers=n_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

###############################################################################

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=rate, momentum=momentum)

train(net, trainloader, testloader, calc_acc,
      criterion, optimizer,
      rate, momentum, n_epochs, acc_file,)

test_acc = accuracy(testloader,net)
with open(total_file,'a') as f:
    f.write(f"{rate:16.4e}{momentum:16.4e}{test_acc:10.4f}\n")
print(f"Final test acc: {test_acc:10.4f}")


