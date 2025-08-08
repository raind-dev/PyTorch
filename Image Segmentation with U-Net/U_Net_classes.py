import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# Complete the U-Net model definition
class SegUNet_L1(nn.Module):
    def __init__(self):
        super(SegUNet_L1, self).__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 64) # 256 --> 254 --> 252
        self.pool1 = nn.MaxPool2d(2) # 252 --> 126
        
        self.bottleneck = ConvBlock(64, 128) # 61 --> 59 --> 57

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Last Output：Binary Segmentation
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # [B, 64, H, W]

        b = self.bottleneck(self.pool1(e1))  # [B, 128, H/2, W/2]

        # Decoder + Skip Connection    
        d1 = self.up1(b)  # [B, 128, H/2, W/2]
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.dec1(d1)
        out = self.final(d1)  # [B, 1, H, W] - Simplified for debugging
        return out

class SegUNet_L2(nn.Module):
    def __init__(self):
        super(SegUNet_L2, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(1, 64) # 256 --> 254 --> 252
        self.pool1 = nn.MaxPool2d(2) # 252 --> 126

        self.enc2 = ConvBlock(64, 128) # 126 --> 124 --> 122
        self.pool2 = nn.MaxPool2d(2) # 122 --> 61
        
        self.bottleneck = ConvBlock(128, 256) # 61 --> 59 --> 57

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Last Output：Binary Segmentation
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)             # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))# [B, 128, H/2, W/2]

        b = self.bottleneck(self.pool2(e2))  # [B, 1024, H/16, W/16]

        # Decoder + Skip Connection
        d2 = self.up2(b)  # [B, 256, H/4, W/4]
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.dec2(d2)
          
        d1 = self.up1(d2)  # [B, 128, H/2, W/2]
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.dec1(d1)
        out = self.final(d1)  # [B, 1, H, W] - Simplified for debugging
        return out

class SegUNet_L3(nn.Module):
    def __init__(self):
        super(SegUNet_L3, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(1, 64) # 256 --> 254 --> 252
        self.pool1 = nn.MaxPool2d(2) # 252 --> 126

        self.enc2 = ConvBlock(64, 128) # 126 --> 124 --> 122
        self.pool2 = nn.MaxPool2d(2) # 122 --> 61
        
        self.enc3 = ConvBlock(128, 256) # 126 --> 124 --> 122
        self.pool3 = nn.MaxPool2d(2) # 122 --> 61
        
        self.bottleneck = ConvBlock(256, 512) # 61 --> 59 --> 57

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Last Output：Binary Segmentation
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)             # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))# [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))# [B, 256, H/4, W/4]

        b = self.bottleneck(self.pool3(e3))  # [B, 512, H/8, W/8]

        # Decoder + Skip Connection
        d3 = self.up3(b)                          # [B, 512, H/8, W/8]
        d3 = torch.cat([d3, e3], dim=1)           # [B, 512+256, H/8, W/8]
        d3 = self.dec3(d3)                        # [B, 256, H/8, W/8]
        
        d2 = self.up2(d3)  # [B, 256, H/4, W/4]
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)  # [B, 128, H/2, W/2]
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d1 = self.dec1(d1)
        out = self.final(d1)  # [B, 1, H, W] - Simplified for debugging
        return out
    
class SegUNet_L4(nn.Module):
    def __init__(self):
        super(SegUNet_L4, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Last Output：Binary Segmentation
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)             # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))# [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))# [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))# [B, 512, H/8, W/8]

        b = self.bottleneck(self.pool4(e4))  # [B, 1024, H/16, W/16]

        # Decoder + Skip Connection
        d4 = self.up4(b)                          # [B, 1024, H/8, W/8]
        d4 = torch.cat([d4, e4], dim=1)           # [B, 1024+512, H/8, W/8]
        d4 = self.dec4(d4)                        # [B, 512, H/8, W/8]

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)  # [B, 2, H, W]
        return out