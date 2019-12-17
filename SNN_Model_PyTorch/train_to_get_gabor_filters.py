# NC trial project:
# load the MNIST images;
# apply the 3x3 fixed conv. kernels
# convert each pixel from every map to a spike train
# extract [12x26x26x1000] matrix


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
from timeit import default_timer as timer
import numpy as np



#now load the training images and apply the kernels to those

#Check if we have a GPU or a CPU
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
print(device)

#Training settings
batch_size = 128
#Get the MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
#initialize the convolutional layer with the Gabor-like filters
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 12, kernel_size=3)
        self.fc = nn.Linear(12*26*26, 10)
        self.lsm = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = x.view(-1, 26 * 26 * 12)
        x = self.fc(x)

        return self.lsm(x)

model = Net()
print("heree!: ", model.conv1.weight.shape)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

epoch_loss= []
def train(epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        count+=1
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 128 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
        if batch_idx % 128 ==0:
            epoch_loss.append(loss.data.item())

test_accuracy = []
def test():
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            output = model(data)

        # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).data.item()
        # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy.append(correct)#(100. * correct / len(test_loader.dataset))


train_epoch_time =[]
test_time = []

for epoch in range(1):
    print("testing first")
    test()
    start = timer()
    train(epoch)
    end = timer()
    train_epoch_time.append((end - start))
    test()
    #now extract the filters
    fw = open('project_data/trained_gabor_filters', 'wb')
    weights =  model.conv1.weight.data
    print(weights.shape)
    pickle.dump(weights, fw)
    fw.close()


