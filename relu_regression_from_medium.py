import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np
import imageio

torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = torch.sin(x) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(1, 200),
        torch.nn.ReLU(),
        # torch.nn.Linear(200, 100),
        # torch.nn.LeakyReLU(),
        # torch.nn.Linear(100, 1),
        torch.nn.Linear(200, 1),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 64
EPOCH = 200

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)

# start training
L = []
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        L.append(loss.item())
        print(epoch, L[-1])

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        # if step == 1:
            # # plot and show learning process
            # plt.cla()
            # ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
            # ax.set_xlabel('Independent variable', fontsize=24)
            # ax.set_ylabel('Dependent variable', fontsize=24)
            # ax.set_xlim(-11.0, 13.0)
            # ax.set_ylim(-1.1, 1.2)
            # ax.scatter(b_x.data.numpy(), b_y.data.numpy(), color = "blue", alpha=0.2)
            # ax.scatter(b_x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
            # ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
                    # fontdict={'size': 24, 'color':  'red'})
            # ax.text(8.8, -0.95, 'Loss = %.4f' % loss.data.numpy(),
                    # fontdict={'size': 24, 'color':  'red'})

            # # Used to return the plot as an image array 
            # # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
            # fig.canvas.draw()       # draw the canvas, cache the renderer
            # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            # image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # my_images.append(image)

    


# save images as a gif    
# imageio.mimsave('./curve_2_model_3_batch.gif', my_images, fps=12)


fig, ax = plt.subplots(figsize=(16,10))
plt.cla()
ax.set_title('Regression', fontsize=35)
ax.set_xlabel('x', fontsize=24)
ax.set_ylabel('y', fontsize=24)
ax.set_xlim(-11.0, 13.0)
ax.set_ylim(-1.1, 1.2)
ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue", alpha=0.2)

x_test = torch.unsqueeze(torch.linspace(-10, 30, 1000), dim=1)  # x data (tensor), shape=(100, 1)
x_test = Variable(x_test)
prediction = net(x_test)     # input x and predict based on x
ax.scatter(x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
plt.savefig('curve_2_model_3_batches.png')
plt.show()
