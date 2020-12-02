'''
Source: https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np
# import imageio

# Training parameters
BATCH_SIZE = 32
EPOCH = 200
HIDDEN = 200

# Function for testing the loss function
def test_bias_during_training(net):
    x_test = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
    x_test = Variable(x_test)
    prediction = net(x_test)     # input x and predict based on x

    return x_test.data.numpy() ,prediction.data.numpy()
    

torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = torch.sin(x) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(1, HIDDEN),
        torch.nn.ReLU(),
        # torch.nn.Linear(HIDDEN, HIDDEN),
        # torch.nn.ReLU(),
        # torch.nn.LeakyReLU(),
        torch.nn.Linear(HIDDEN, 1),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)

# start training
L = []
x_bias_test = []
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        L.append(loss.item())
        print(epoch, L[-1])
        if epoch % 100 == 99:
            x_bias_test.append(test_bias_during_training(net)[1])

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

    if epoch % 10 == 0:
        x_bias_test.append(test_bias_during_training(net)[1])
        print(epoch)


# Plotting results
fig, ax = plt.subplots(figsize=(16,10))
plt.cla()
ax.set_title('Regression', fontsize=35)
ax.set_xlabel('x', fontsize=24)
ax.set_ylabel('y', fontsize=24)
ax.set_xlim(-11.0, 13.0)
ax.set_ylim(-1.1, 1.2)
ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue", alpha=0.2)

x_test = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
x_test = Variable(x_test)
prediction = net(x_test)     # input x and predict based on x
ax.scatter(x_test.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
# plt.savefig('curve_2_model_3_batches.png')
plt.show()

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.plot(x.data.numpy(), x_bias_test[i])
    plt.plot(x.data.numpy(), y.data.numpy())

plt.show()
