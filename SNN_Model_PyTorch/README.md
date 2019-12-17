This is a Pytorch implementation of an SNN model, which is described in the article ``Spiking neural networks for handwritten digit recognition—Supervised learning and network optimization'', Shruti R. Kulkarni, Bipin Rajendran ( Neural Networks, 2018). 
In particular, the forward pass of training the model is analyzed in detail, including the pixel-to-spikes conversion method, the generation of input currents, and the generation of spikes in the output layer from the input currents from the hidden layer. 
Python methods that implement each of these steps are provided. The pdf report provides details about each of the scripts and their function. 
Furthermore, a modified version of the forward step is discussed, which reduces the time needed for a single image forward pass, while producing similar spike-trains for the hidden-layer neurons. 
Thus, the contributions of this project are the detailed analysis and explanation of the forward pass of the SNN model, 
the Python/PyTorch implementation of the forward step of the SNN model 
(the scripts are self-contained and the user is required to just have installed the latest PyTorch version), 
and a variation of the forward steps which is significantly faster than the original forward method. 

In addition, a Jupyter notebook explaning most of the forward pass steps is provided. 
