import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
from numpy import transpose


def weights_init(m):
    # custom weights initialization called on ``netG`` and ``netD``
    # Initialize the weights of the Generator and Discriminator
    # with numbers sampled from a normal distirbution (mean and
    # std are passed as 0.0 and 0.02)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_gd_loss(G_losses, D_losses, output_name):
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
