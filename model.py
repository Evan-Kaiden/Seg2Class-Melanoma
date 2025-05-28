import torch
import torch.nn as nn
import torch.nn.functional as F



class down(nn.Module):
    def __init__(self, in_channels, out_channels=None, pool=True):
        super(down, self).__init__()
        self.pool = pool

        if out_channels is None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(p=0.3),
                nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, padding=1),
                nn.ReLU(),  
                nn.Dropout2d(p=0.3), 
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(p=0.3),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),   
                nn.Dropout2d(p=0.3),
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
            nn.Dropout2d(p=0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
        )
        self.conv_none = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
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


        # return mask as well as residual outputs
        return out, d0, d1, d2, d3
    

class Seg2Class(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Seg2Class, self).__init__()

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (B, C, 1, 1)
            nn.Flatten(),                  # (B, C)
            nn.Linear(128 + 1, n_classes), # (B, n_classes)
            nn.Dropout(p=0.3)
        )

        self.unet = Unet(in_channels, n_classes=1)

    def forward(self, x):
        mask, d0, d1, d2, d3 = self.unet(x)
        
        # downsample the mask 
        B, C, H, W = d1.shape
        scaled_mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

        # Concat
        d1_mask = torch.concat([d1, scaled_mask], dim = 1) # (B x C+1 x H x W)

        # Predict
        classification = self.mlp(d1_mask)

        return classification, mask



        

        



x = torch.randn(1, 3, 256, 256) 

model = Seg2Class(in_channels=3, n_classes=10)

model.eval()

with torch.no_grad():
    out, mask = model(x)

# Print the shapes
print(f"Input shape:      {x.shape}")
print(f"Output Shape:     {out.shape}")
print(f"Mask shape:         {mask.shape}")