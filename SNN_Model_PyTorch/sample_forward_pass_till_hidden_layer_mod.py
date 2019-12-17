#I will first do the conversion on the testing data set as I first wanna do/test the forward pass
#so, for every image in the test set, convert every pixel from the image to spike train
#thus a 28x28 image => 28x28x1000
#next apply the conv. filters spatially, namelly for each t=1...1000 apply filter on the 28x28 ``time stamp''

from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import pickle
import numpy as np
import data_helper_functions as dhf
import matplotlib.pyplot as plt
import from_current_to_spikes as cs
import math
#read the predefined Gabor-like conv filters
path="kernels_3x3.csv"
kernels = pickle.load(open('project_data/trained_gabor_filters', 'rb'))
print("here: ", kernels.shape)
#kernels = dhf.load_predefined_kernels(path)
#kernels= torch.tensor(kernels).float()

#Check if we have a GPU or a CPU
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
print(device)

#Training settings
batch_size = 1000
#Get the MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               #transform=transforms.ToTensor(),
                               download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor()
                              )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
#initialize the convolutional layer with the Gabor-like filters
class Net(nn.Module):

    def __init__(self, gabor_filters):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, bias=False)
        #print("weights shape: ", self.conv1.weight.data.shape)
        self.conv1.weight.data = gabor_filters
        #self.conv1.weight.data = self.conv1.weight.data[:,None, :]
        print("weights shape: ", self.conv1.weight.data.shape)

    def forward(self, x):
        x = self.conv1(x)
        flattened = x.view(-1,12*26*26)
        return x, flattened

model = Net(torch.FloatTensor(kernels))
#model.double()
labels = dhf.get_all_labels(test_loader)
data = dhf.get_all_data(test_loader)

groups = dhf.get_groups_of_class_indices(labels, 10)
considered_class = 9
data_to_consider = data[groups[considered_class]]

#take just NUM_IMGS images
NUM_IMGS = 100
data_to_consider = data_to_consider[:NUM_IMGS]
labels_to_consider = labels[groups[considered_class]]

#load the spike trains for the pixels
pixel_spikes = pickle.load(open('project_data/pixel_spike_trains', 'rb'))
print(pixel_spikes.shape)

#for each image, get a tensor of size 1000x28x28 [for each pixel hold the spike-train, essentially]
spikes_tensors = torch.empty((len(data_to_consider),1000,26,26))

#now here, I first apply the model on the data as usual and get the 12 26x26 feature maps
f1, flattened = model(torch.tensor(data_to_consider))
print("checkpoint: ", flattened.shape)

#now, every value in flattened is a constant current input;
NUM_IMGS2=1
for img in range(NUM_IMGS2):
    input_current = np.transpose(flattened[img].detach().numpy())
    input_current = np.reshape(input_current, [8112,1])
    input_current = np.repeat(input_current, 1000, axis=1)
    decay=0.966
    for i in range(input_current.shape[1]-1, -1, -1):
        input_current[:,i]=input_current[:,i]*math.pow(decay,i)

    Y_spk = cs.convert_current_to_spikes(input_current)

#now visualize the spikes of the image
Y_spk_to_viz = Y_spk.transpose().reshape([1000,12,26,26])

#visualize all 12 kernels in one figure

fig, ax = plt.subplots(4,3, dpi=120)
row_count=0
col_count=0
for kernel in range(12):
    digit_to_vis_for_ker0 = Y_spk_to_viz[:,kernel,:,:]
    print("digit to vis shape: ", digit_to_vis_for_ker0.shape)
    count = np.empty((26,26))

    for i in range(26):
        for j in range(26):
            s = np.sum(digit_to_vis_for_ker0[:,i,j])
            if s<=27:
                s=0
            count[i,j] = np.sum(digit_to_vis_for_ker0[:,i,j])
    print(count)

    #now, visualize count
    ax[row_count][col_count].imshow(count, vmin=np.min(count), vmax=np.max(count))
    col_count+=1
    if col_count==3:
        row_count+=1
        col_count=0

plt.show()