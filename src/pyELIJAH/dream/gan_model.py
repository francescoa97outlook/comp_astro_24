import random
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from pyELIJAH.detection.machine_learning.Dataset import DatasetTrainDev
from pyELIJAH.dream.Generator import Generator
from pyELIJAH.dream.Discriminator import Discriminator
from pyELIJAH.dream.gan_utils import weights_init, plot_gd_loss, plot_real_fake, plot_lc


def gan_model(
        input_data_folder, output_folder, params_list
):
    """
    Function that implement a Generative Adversarial Network to create new 
    light curve images from a train dataset. The only accepted image size
    is 56 pixels by 56 pixels and only 1 channel.

    Args:
        input_data_folder (str): Path to the folder containing the training data
        output_folder (str): Path to the folder where to save the output data
        params_list (list of parameters obj): Object containing list of model parameters
    """
    #
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    #
    for params in params_list:
        # Retrieve data information
        data_object = DatasetTrainDev(
            input_data_folder, "gan",
            params.get("filename_train"), params.get("filename_dev"),
            params.get("array_lenght"), params.get("image_size")
        )
        X_train, Y_train = data_object.get_train()
        # Transform the Y from boolean to integers
        Y_train = Y_train.astype(int)
        # Transform to torch tensors
        X_train = torch.Tensor(X_train)
        Y_train = torch.Tensor(Y_train)
        # From NHWC to NCHW
        # N=number, H=height, W=width, C=channels
        X_train = X_train.permute(0, 3, 1, 2)
        # Create the dataset
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        # Define some inputs
        # Number of workers for dataloader
        workers = 2
        # Batch size during training
        batch_size = params.get("batch_size")
        # Number of channels in the training images. For color images this is 3
        nc = 1
        # Size of z latent vector (i.e. size of generator input)
        nz = params.get("nz")
        # Size of feature maps in generator
        ngf = params.get("ngf")
        # Size of feature maps in discriminator
        ndf = params.get("ndf")
        # Number of training epochs
        num_epochs = params.get("epoch")
        # Learning rate for optimizers
        lr = 0.0002
        # B eta1 hyperparameter for Adam optimizers
        beta1 = 0.5
        # Number of GPUs available. Use 0 for CPU mode.
        ngpu = params.get("ngpu")
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=workers)

        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Create the generator and discriminator
        netG = Generator(nz, nc, ngf, ngpu).to(device)
        netD = Discriminator(nc, ndf, ngpu).to(device)

        # Handle multi-GPU if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the ``weights_init`` function to randomly initialize all weights
        #  to ``mean=0``, ``stdev=0.02``.
        netG.apply(weights_init)
        netD.apply(weights_init)

        # Print the models
        print(netG)
        print(netD)

        # Initialize the ``BCELoss`` function
        # Binaty Cross-Entropy loss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # Training Loop
        # Lists to keep track of progress
        lc_list = np.zeros(shape=(1,1,56,56))
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float,
                                   device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed)
                # with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                # fake labels are real for generator cost
                label.fill_(real_label)
                # Since we just updated D, perform another forward pass
                # of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    lc_list = np.append(lc_list, fake, axis=0)

                iters += 1

        # Plot generator and discriminator loss
        current_datetime = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        output_loss = str(Path(
            output_folder, f"plot_gan_losses_{current_datetime}.png"
        ))
        plot_gd_loss(G_losses, D_losses, output_loss)

        # Save the list of generated images
        output_list = str(Path(
            output_folder, f"list_gan_fake_{current_datetime}.npy"
        ))
        np.save(output_list, img_list)

        # Show an animation of the generated images
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000,
                                        repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))
        output_image = str(Path(
            output_folder, f"plot_gan_{current_datetime}.png"
        ))

        # Plot real and generated images
        plot_real_fake(real_batch, img_list, device, output_image)

        # To visualize the light curves from both real and generated image
        # From NCHW to NHWC
        X_train = X_train.permute(0, 2, 3, 1)
        to_delete = []
        for i in range(len(Y_train)):
            if Y_train[i] == 0:
                to_delete.append(i)
        X_train = np.delete(X_train, to_delete, axis=0)
        # lc_list = lc_list[:3136]
        output_lc_image = str(Path(
            output_folder, f"plot_lc_gan_{current_datetime}.png"
        ))
        # Plot the light curves from a real and a generated image
        plot_lc(X_train, lc_list,output_lc_image)
