import torch
import torch.nn as nn



class down(nn.Module):
    def __init__(self, in_channels):
        super(down, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, padding=1),
            nn.ReLU(),   
        )

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    def forward(self, x):
        pre_pooled_x = self.conv(x)
        pooled_x = self.pool(pre_pooled_x)

        return pooled_x, pre_pooled_x 
    

class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2,2), stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = torch.concat([x, skip_connection], dim=1)

        x = self.conv(x)

        return x


class Unet(nn.Module):
    def __init__(self, input_shape):
        pass
