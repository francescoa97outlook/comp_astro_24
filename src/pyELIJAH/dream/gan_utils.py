import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn
from numpy import transpose
from scipy import ndimage


def weights_init(m):
    """
    Function that initialize the weight of the Generator and Discriminator
    with number sampled from a normal distribution (mean and std are
    passed as 0.0 and 0.02)

    Args:
        m: layer considered for the weight
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_gd_loss(G_losses, D_losses, output_name):
    """
    Function that plots the loss function of the Generator and Discriminator

    Args:
        G_losses (array): array of the loss function values of the Generator
        D_losses (array): array of the loss function values of the Discriminator
        output_name (str): the string used to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Generator and Discriminator Loss During Training")
    ax.plot(G_losses, label="G")
    ax.plot(D_losses, label="D")
    ax.set_xlabel("iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(output_name)
    plt.show()
    plt.close(fig)


def plot_real_fake(real_batch, img_list, device, output_name):
    """
    Function to visualize side by side a selection of real and generated images

    Args:
        real_batch: batch of real images in tensor format
        img_list: list of arrays representing generated images
        device: the device (CUP of GPU) used
        output_name: the string used to save the plot
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    # Plot the real images
    axs[0].axis("off")
    axs[0].set_title("Real Images")
    axs[0].imshow(transpose(vutils.make_grid(
        real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    axs[1].axis("off")
    axs[1].set_title("Fake Images")
    axs[1].imshow(transpose(img_list[-1], (1, 2, 0)))

    plt.savefig(output_name)
    plt.show()
    plt.close(fig)

def plot_lc(real_batch, img_list, output_name):
    """
    Function to visualize the light curve from a real and a generated image side by side

    Args:
        real_batch: batch of real images in tensor format
        img_list: list of arrays representing generated images
        output_name: the string used to save the plot
    """

    image_size = 56
    # Convert and create the arrays
    img_list = np.array(img_list)
    real_batch = np.array(real_batch)
    arr_real = np.empty(shape=(len(real_batch), image_size*image_size))
    arr_img = np.empty(shape=(len(img_list), image_size*image_size))
    # Reshape as single row of fluxes
    for i, _ in enumerate(real_batch):
        arr_real[i] = real_batch[i, :, :, 0].reshape(image_size*image_size)
    for i, _ in enumerate(img_list):
        arr_img[i] = img_list[i, 0, :, :].reshape(image_size*image_size)
    # Apply gaussian filter (smooth)
    arr_img = ndimage.gaussian_filter(arr_img, sigma=10)

    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    axs[0].plot(arr_real[0,:])
    axs[0].set_ylabel("Normalized Flux")
    axs[0].set_title("Light curve from a real image")

    axs[1].plot(arr_img[-1, :])
    axs[1].set_ylabel("Flux")
    axs[1].set_title("Light curve from a generated image")

    plt.savefig(output_name)
