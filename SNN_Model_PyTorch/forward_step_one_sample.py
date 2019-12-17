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
import from_current_to_spikes as cs
import from_hidden_spikes_to_current_for_outputs as sk
import from_current_to_spikes as ck


#read the predefined Gabor-like conv filters
path="kernels_3x3.csv"
kernels = dhf.load_predefined_kernels(path)
kernels= torch.tensor(kernels).float()

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
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3)
        #print("weights shape: ", self.conv1.weight.data.shape)
        self.conv1.weight.data = gabor_filters
        self.conv1.weight.data = self.conv1.weight.data[:,None, :]
        #print("weights shape: ", self.conv1.weight.data.shape)

    def forward(self, x):
        x = self.conv1(x)
        flattened = x.view(-1,12*26*26)
        return x, flattened

model = Net(torch.FloatTensor(kernels))
model.double()
labels = dhf.get_all_labels(test_loader)
data = dhf.get_all_data(test_loader)

groups = dhf.get_groups_of_class_indices(labels, 10)
considered_class = 9
data_to_consider = data[groups[considered_class]]

#take just NUM_IMGS images
NUM_IMGS = 1
data_to_consider = data_to_consider[:NUM_IMGS]
labels_to_consider = labels[groups[considered_class]]

#load the spike trains for the pixels
pixel_spikes = pickle.load(open('project_data/pixel_spike_trains', 'rb'))
print("Spike trains for each pixel are loaded, shape is: ", pixel_spikes.shape)

#for each image, get a tensor of size 1000x28x28 [for each pixel hold the spike-train, essentially]
spikes_tensors = torch.empty((len(data_to_consider),1000,28,28))

for img in range(len(data_to_consider)):
    for pix_i in range(28):
        for pix_j in range(28):
            spikes_tensors[img, :,pix_i,pix_j] = \
                torch.tensor(pixel_spikes[int(data_to_consider[img].squeeze()[pix_i,pix_j]*255)])

print("The input image has 28x28 pixels which are now spike trains, thus shape of img tensor is: ", spikes_tensors.shape)

#evaluate the model on each ``time-snapshot'',
#so essentially for every image we have a batch of 1000 time-snapshots of spatial dim. 28x28 that we convolve
model.eval()
forwarded_digits = []
for img in range(NUM_IMGS):
    spikes_tensor_to_consider =spikes_tensors[img][:,None,:].double()
    f1, flattened = model(spikes_tensor_to_consider)
    forwarded_digits.append(f1.detach())

#from the convolution we get the input current for the hidden neurons
i_in = np.transpose(forwarded_digits[0].reshape((1000,8112)).numpy())
print("The current, that is input to the 8112 hidden neurons is calculated by the conv. kernels and has shape: ", i_in.shape)

#now, convert the current to spikes
Y_spk = cs.convert_current_to_spikes(i_in)
print("We converted the current to spikes; the spike tensor has shape: ", Y_spk.shape)


#load the weights, initialized arbitrary now
weights = np.random.normal(0, 1, (8112,10))
weights = np.transpose(weights)

print("Weights, currently initialized as Gaussian(0,1) loaded, shape:", weights.shape)

output_spikes = np.empty((10,1000))

for i in range(NUM_IMGS):
    i_in = sk.from_spikes_to_current(Y_spk, np.abs(weights))

print("We have converted the spikes of the hidden neurons to current, shape is: ", i_in.shape)
#now, get the spikes of the output neurons
output_spikes = ck.convert_current_to_spikes(i_in)
print("We have inputed the current of the hidden neurons to the output neurons and we calculated the spike trains of the output neurons, shape: ", output_spikes.shape)
sums=np.empty((10))
for k in range(10):
    sums[k]=sum(output_spikes[k,:])
print("Based on which output neuron had the highest spike count: ")
print("Guessed label is: ", np.argmax(sums))
print("True label is: ", labels_to_consider[0])
