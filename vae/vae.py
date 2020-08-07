from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

from scipy.io import loadmat

parser = argparse.ArgumentParser(description='VAE COIL20 Example')

parser.add_argument('--dataset', type=str, default='toy',
                    help='Which dataset to load: "toy" (default), "mnist" or "coil20" from .mat file')
parser.add_argument('--latent-space-dim', type=int, default=10, metavar='D',
                    help='Size of the latent/code space')
parser.add_argument('--hidden-state-dim', type=int, default=400, metavar='h',
                    help='Size of the hidden space')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='lr',
                    help='Learning rate for Adam optimizer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

'''
Loading dataset
'''

#TODO put below in function
if args.dataset == 'toy':
    ''' Uniform square with 10000 data points '''
    # dataset = np.random.rand(10000, 2)/2

    ''' Normal distribution with 10000 data points '''
    dataset = normalize(np.random.randn(10000, 2)/2, axis=0)
    data = torch.from_numpy(dataset)
    n, d = dataset.shape

    toy_labels = torch.from_numpy(np.ones(n))

    training_data = torch.utils.data.TensorDataset(data.float(), toy_labels)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

elif args.dataset == 'coil20':
    dataset = loadmat('./coil20.mat')
    n, d = dataset['data'].shape
    data = torch.from_numpy(normalize(dataset['data']))
    labels = torch.from_numpy(dataset['objlabel'])

    training_data = torch.utils.data.TensorDataset(data.float(), labels)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

elif args.dataset == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    d = 784

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Latent space dimension
        latent_space_dim = args.latent_space_dim
        hidden_state_dim = args.hidden_state_dim

        self.fc1 = nn.Linear(d, hidden_state_dim)
        self.fc21 = nn.Linear(hidden_state_dim, latent_space_dim)
        self.fc22 = nn.Linear(hidden_state_dim, latent_space_dim)
        self.fc3 = nn.Linear(latent_space_dim, hidden_state_dim)
        self.fc4 = nn.Linear(hidden_state_dim, d)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
        # return self.fc4(h3)

    def forward(self, x):
        ''' The forward pass '''
        mu, logvar = self.encode(x.view(-1, d)) # View is basically reshaping
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, d), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            # if i == 0:
                # n = min(data.size(0), 8)
                # comparison = torch.cat([data[:n],
                                      # recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                # save_image(comparison.cpu(),
                         # 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def sample_and_plot():
    with torch.no_grad():
        num_samples = 1000
        sample = torch.randn(num_samples, args.latent_space_dim).to(device)
        sample = model.decode(sample).cpu()

    # Plotting some samples
    plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1])
    plt.scatter(sample[:, 0], sample[:, 1])
    plt.legend(['dataset', 'sample'])
    plt.show()


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test(epoch)




    # Visualizing training data
    # train_data = train_loader.dataset.train_data
    # train_data = torch.reshape(train_data,[ 60000, 784])
    # train_data_encoded = model.encode(train_data.float())
    # train_data_encoded = train_data_encoded[0]

    # # Detaching the training set
    # tde = train_data_encoded.detach()

    # # Create plot
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax = fig.gca(projection='3d')

    # # plotting the 2000 first training data points
    # ax.scatter(tde[0:2000,0],tde[0:2000,1],tde[0:2000,2], c=train_loader.dataset.train_labels[0:2000])
    # # ax.scatter(tde[0:2000,1],tde[0:2000,2],tde[0:2000,3], c=train_loader.dataset.train_labels[0:2000])
    # fig.show()


