import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
import time


# Class for the double convolutional steps in the U-Net model.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # 2D convolutional layer with kernel size = 3, stride = 1 and
            # zero padding = 1, added to make it an same convolution.
            # Bias set to False since we use batch normalization.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # Add batch normalization to speed up convergence
            nn.BatchNorm2d(out_channels),
            # ReLU activation, inplace means that the input is modified directly,
            # used to save memory.
            nn.ReLU(inplace=True),
            # 2D convolutional layer with kernel size = 3, stride = 1 and
            # zero padding = 1, added to make it an same convolution.
            # Bias set to False since we use batch normalization.
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # Add batch normalization to speed up convergence
            nn.BatchNorm2d(out_channels),
            # ReLU activation, inplace means that the input is modified directly,
            # used to save memory.
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

# Class for the complete light U-Net architecture.
class MimoUNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, scaling=1, ensemble_size=1):
        super(MimoUNET, self).__init__()
        # List of all modules for the up sampling part.
        self.ups = nn.ModuleList()
        # List of all modules for the down sampling part.
        self.downs = nn.ModuleList()
        # Max pooling layer, does not have any parameters.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # The different dimmensions for the feature maps.
        feature_map_dims = [64, 128, 256, 512]
        channels = in_channels * ensemble_size

        # Add all modules to the down sampling part
        # Go thorugh all dims and add the convolutional layers
        for dim in feature_map_dims:
            # Add the double convolution step for the dim
            self.downs.append(DoubleConv(channels, dim))
            channels = dim

        # Add all modules to the up sampling part
        for dim in reversed(feature_map_dims):
            # Add the Deconvolution layer
            self.ups.append(
                nn.ConvTranspose2d(
                    dim*2, dim, kernel_size=2, stride=2,
                )
            )
            # Add the double convolution
            self.ups.append(DoubleConv(dim*2, dim))

        # Initialize the bottom layer and the final output step.
        self.width = 120
        self.height = 160
        self.scale = scaling
        self.num_ens = ensemble_size
        self.bottom_layer = DoubleConv(feature_map_dims[-1], feature_map_dims[-1]*2)
        self.final_conv = nn.Conv2d(feature_map_dims[0], out_channels, kernel_size=1)
        self.linear1 = nn.Linear(out_channels * self.width * self.height, 32*ensemble_size)
        self.linear2 = nn.Linear(32*ensemble_size, 8*ensemble_size)
        self.num_out = out_channels

    def forward(self, input):
        skip_connections = []

        # Feed the input through the down sampling part
        for down in self.downs:
            input = down(input)
            skip_connections.append(input)
            input = self.pool(input)

        # Apply the bottom layer to the input
        input = self.bottom_layer(input)

        # Reverse the skip connections, so the bottom one is first
        skip_connections = skip_connections[::-1]

        # Feed the input through the up sampling part
        for idx in range(0, len(self.ups), 2):
            # Apply the up sampling layer
            input = self.ups[idx](input)
            skip_connection = skip_connections[idx//2]

            # Reshape the input if not dividable by 16
            if input.shape != skip_connection.shape:
                input = TF.resize(input, size=skip_connection.shape[2:])

            # Combine the up sample result and the skip connection.
            concat_skip = torch.cat((skip_connection, input), dim=1)
            input = self.ups[idx+1](concat_skip)

        input = self.final_conv(input)
        input = input.view(input.shape[0], 1, self.num_out * self.height * self.width)
        input = self.linear1(input)
        # input = self.linear2(input)

        # Get the mask from the final feature map.
        if self.scale == 1:
            return torch.sigmoid(self.linear2(input).view(input.shape[0], 1, self.num_ens, 8))
        else:
            return self.linear2(input).view(input.shape[0], 1, self.num_ens, 8)

def main():
    num_ens = 2
    x = torch.randn((4, 3*num_ens, 120, 160))
    model = MimoUNET(in_channels=3, out_channels=1, ensemble_size=num_ens)
    start = time.time()
    y = model(x)
    end = time.time()
    print(f'Time of forward pass: {end - start}s')

if __name__ == "__main__":
    main()