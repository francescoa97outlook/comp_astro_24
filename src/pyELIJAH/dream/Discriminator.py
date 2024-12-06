import torch.nn as nn


class Discriminator(nn.Module):
    """
    Class that describe the Discriminator model
    """

    def __init__(self, nc, ndf, ngpu):
        """
        This init the Discriminator object

        Args:
            nc (int): number of channels in the training images
            ndf (int): size of feature map in the discriminator
            ngpu (int): number of available GPUs
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu
        #
        # Build model
        self.main = nn.Sequential(
            # input is ``(nc) x 56 x 56``
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 28 x 28``
            nn.Conv2d(in_channels=self.ndf, out_channels=self.ndf * 2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 14 x 14``
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf * 4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 7 x 7``
            nn.Conv2d(in_channels=self.ndf * 4, out_channels=self.ndf * 8,
                      kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(in_channels=self.ndf * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0,
                      bias=False),
            # state size. ``(1) x 1 x 1``
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
