import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt




# Vanilla Autoencoder
class VanillaAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.ReLU(True),
            nn.Linear(500, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x




# Variational AutoEncoder
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(),
            nn.Linear(500, latent_dim * 2))

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 28 * 28),
            nn.Sigmoid())

    def reparameterize(self, x):
        mean, logvar = x.split(latent_dim, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.reparameterize(x)
        x = self.decoder(x)
        return x

def loss_fn(reconstructed, original, mu, logvar, lower_bound, batch_size):
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed, original, reduction='sum')
    kl_divergence = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    lower_bound.append(reconstruction_loss/batch_size)
    return reconstruction_loss + kl_divergence





def test_model(latent_dim, batch_size, num_epochs, learning_rate):

    ############################
    ### Download the dataset ###
    ############################

    train_dataset = dsets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='data/', train=False, transform=transforms.ToTensor())

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    ########################
    ### Train the models ###
    ########################

    # Vanilla Autoencoder
    vanilla_model = VanillaAutoencoder(latent_dim)
    vanilla_log_likelihood = []
    vanilla_optimizer = optim.Adam(vanilla_model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Train the model
    print('Training of the Vanilla AE \n')
    for epoch in range(num_epochs):
        for data in train_loader:
            images, _ = data
            images = images.view(images.size(0), -1)
            vanilla_optimizer.zero_grad()
            reconstructed = vanilla_model(images)
            loss = criterion(reconstructed, images)
            vanilla_log_likelihood.append(loss)
            loss.backward()
            vanilla_optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean([vanilla_log_likelihood[i].item() for i in range(len(vanilla_log_likelihood))]):.4f}')

    # Variational Autoencoder
    variational_model = VAE(latent_dim)
    variational_log_likelihood = []
    variational_lower_bound = []
    variational_optimizer = optim.Adam(variational_model.parameters(), lr=learning_rate)

    # Train the model
    print('\n Training of the Variational AE \n')
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(train_loader):
            images = images[1].view(1, 28 * 28)
            reconstructed = variational_model(images)
            mu, logvar = variational_model.encoder(images).split(latent_dim, dim=1)
            loss = loss_fn(reconstructed, images, mu, logvar, variational_log_likelihood, batch_size)
            variational_lower_bound.append(loss)
            variational_optimizer.zero_grad()
            loss.backward()
            variational_optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean([variational_lower_bound[i].item() for i in range(len(variational_lower_bound))]):.4f}')

    #####################################
    ### Test and visualize the models ###
    #####################################

    # Plot a set of images, and the reconstructed images given by the Vanilla and the Variational AE
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        # Get a batch of test images
        images = images.view(images.size(0), -1)
        nb_images_to_show = min(10, images.size(0))

        # Create a figure with a grid of subplots
        fig, axs = plt.subplots(3, nb_images_to_show, figsize=(20, 10))
        fig.tight_layout()
        for i in range(nb_images_to_show):
            original = images[i].view(28, 28)
            vanilla_reconstructed = vanilla_model(images[i].view(1, -1)).view(28, 28)
            variational_reconstructed = variational_model(images[i].view(1, -1)).view(28, 28)
            # Display the images in the subplots
            axs[0, i].imshow(original, cmap='gray')
            axs[1, i].imshow(vanilla_reconstructed, cmap='gray')
            axs[2, i].imshow(variational_reconstructed, cmap='gray')

        plt.show()

    # Compute and plot the log_likelihoods for the test set
    with torch.no_grad():

        vanilla_log_likelihood = []
        variational_log_likelihood = []
        variational_lowerbound = []
        for i, (images, _) in enumerate(test_loader):
            images = images[1].view(1, 28 * 28)
            vanilla_reconstructed = vanilla_model(images)
            variational_reconstructed = variational_model(images)
            mu, logvar = variational_model.encoder(images).split(latent_dim, dim=1)
            vanilla_loss = criterion(vanilla_reconstructed, images)
            vanilla_log_likelihood.append(vanilla_loss)
            variational_loss = loss_fn(variational_reconstructed, images, mu, logvar, variational_log_likelihood, batch_size)
            variational_lowerbound.append(variational_loss)

        mean_vanilla_log_likelihood = [vanilla_log_likelihood[0]]
        for i in range(1, len(vanilla_log_likelihood)):
            mean_vanilla_log_likelihood.append((1/i)*((i-1)*mean_vanilla_log_likelihood[-1] + vanilla_log_likelihood[i]))

        mean_variational_log_likelihood = [variational_log_likelihood[0]]
        for i in range(1, len(variational_log_likelihood)):
            mean_variational_log_likelihood.append((1/i)*((i-1)*mean_variational_log_likelihood[-1] + variational_log_likelihood[i]))

        mean_variational_lowerbound = [variational_lowerbound[0]]
        for i in range(1, len(variational_lowerbound)):
            mean_variational_lowerbound.append((1/i)*((i-1)*mean_variational_lowerbound[-1] + variational_lowerbound[i]))

        plt.plot(np.negative(mean_vanilla_log_likelihood), color='blue', label='Vanilla Autoencoder (test)')
        plt.plot(np.negative(mean_variational_log_likelihood), color='red', label='VAE (test)')
        plt.legend()
        plt.title('Log-likelihood for the Vanilla and the Variational AE')
        plt.show()

        plt.plot(mean_variational_lowerbound, color='blue', label='Variational lowerbound (train)')
        plt.legend()
        plt.title('Variational lowerbound of the VAE during the training')
        plt.show()

    ############################################
    ### Plot the latent space in the 2D case ###
    ############################################

    if latent_dim == 2:
        n = 15
        figsize = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-1, 1, n)
        grid_y = np.linspace(-1, 1, n)

        for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([xi, yi])
            x_decoded = variational_model.decoder(z_sample.float())
            digit = x_decoded.detach().numpy().reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

        plt.figure(figsize=(figsize, figsize))
        plt.imshow(figure, cmap="Greys_r")
        plt.show()

        return variational_model, vanilla_model




batch_size = 100
num_epochs = 40
learning_rate = 1e-3
latent_dim = 3

variational_model, vanilla_model = test_model(latent_dim, batch_size, num_epochs, learning_rate)