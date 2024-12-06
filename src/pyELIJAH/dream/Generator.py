import torch.nn as nn


class Generator(nn.Module):
    """
    Class that describe the Generator model
    """

    def __init__(self, nz, nc, ngf, ngpu):
        """
        This init the Generator object

        Args:
            nz (int): size of z latent vector (i.e. size of generator input)
            nc (int): number of channels in the training images
            ngf (int): size of feature map in the generator
            ngpu (int): number of available GPUs
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.ngpu = ngpu
        #
        # Build model
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.nz,
                               out_channels=self.ngf * 4,
                               kernel_size=7, stride=1,
                               padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 7 x 7``
            nn.ConvTranspose2d(in_channels=self.ngf * 4,
                               out_channels=self.ngf * 2,
                               kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 14 x 14``
            nn.ConvTranspose2d(in_channels=self.ngf * 2,
                               out_channels=self.ngf,
                               kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 28 x 28``
            nn.ConvTranspose2d(in_channels=self.ngf,
                               out_channels=self.nc,
                               kernel_size=4, stride=2,
                               padding=1, bias=False),
            # state size. ``(nc) x 56 x 56``
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
