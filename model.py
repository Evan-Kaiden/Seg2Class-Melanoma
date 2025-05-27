import torch
import torch.nn as nn



class down(nn.Module):
    def __init__(self, in_channels, out_channels=None, pool=True):
        super(down, self).__init__()
        self.pool = pool

        if out_channels is None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, padding=1),
                nn.ReLU(),   
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),   
            )
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    def forward(self, x):
        pre_pooled_x = self.conv(x)
        if self.pool:
            pooled_x = self.max_pool(pre_pooled_x)
            return pooled_x

        return pre_pooled_x 
    

class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2,2), stride=2)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_none = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip_connection=None):
        x = self.deconv(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
            x = self.conv_cat(x)
        else:
            x = self.conv_none(x)

        return x


class Unet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Unet, self).__init__()

        self.down0 = down(in_channels, 64)
        self.down1 = down(64)
        self.down2 = down(128)
        self.down3 = down(256)

        self.up0 = up(512, 256)
        self.up1 = up(256, 128)
        self.up2 = up(128, 64)
        self.up3 = up(64, 32)

        self.out = nn.Conv2d(32, n_classes, kernel_size=(1,1))

    def forward(self, x):
        
        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)


        u0 = self.up0(d3, d2)
        u1 = self.up1(u0, d1)
        u2 = self.up2(u1, d0)
        u3 = self.up3(u2, None)
        out = self.out(u3)

        return out
    

def test_unet_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Unet(in_channels=3, n_classes=2).to(device)
    
    # Create a random input tensor with batch size 1, 3 channels, 256x256
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # Forward pass
    output = model(input_tensor)

    print("Output shape:", output.shape)

    # Check output shape (should match batch size and be [1, n_classes, H, W])
    assert output.shape == (1, 2, 256, 256), f"Unexpected output shape: {output.shape}"
    print("Unet forward pass successful.")

