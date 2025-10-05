import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    """
    Improved U-Net with skip connections and adaptive size handling
    for better feature preservation at full resolution
    """
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self._block(in_channels, features, normalize=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._block(features, features*2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self._block(features*2, features*4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = self._block(features*4, features*8)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._block(features*8, features*8)
        
        # Decoder (Upsampling) with skip connections
        self.upconv4 = nn.ConvTranspose2d(features*8, features*8, 2, stride=2)
        self.dec4 = self._block(features*16, features*4)  # *16 due to skip
        
        self.upconv3 = nn.ConvTranspose2d(features*4, features*4, 2, stride=2)
        self.dec3 = self._block(features*8, features*2)
        
        self.upconv2 = nn.ConvTranspose2d(features*2, features*2, 2, stride=2)
        self.dec2 = self._block(features*4, features)
        
        self.upconv1 = nn.ConvTranspose2d(features, features, 2, stride=2)
        self.dec1 = self._block(features*2, features)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Tanh()
        )
        
    def _block(self, in_c, out_c, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
        ])
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with skip connections and size matching
        # Upsample and match size for concatenation
        u4 = self.upconv4(b)
        u4 = self._match_size(u4, e4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        
        u3 = self.upconv3(d4)
        u3 = self._match_size(u3, e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.upconv2(d3)
        u2 = self._match_size(u2, e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.upconv1(d2)
        u1 = self._match_size(u1, e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.final(d1)
    
    def _match_size(self, tensor, target):
        """Match tensor size to target size using interpolation if needed"""
        if tensor.shape[2:] != target.shape[2:]:
            tensor = F.interpolate(tensor, size=target.shape[2:], mode='bilinear', align_corners=True)
        return tensor


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for high-frequency detail preservation
    """
    def __init__(self, in_channels=1, features=64):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: 600x600 or any size
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features, features*2, 4, 2, 1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features*2, features*4, 4, 2, 1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features*4, features*8, 4, 1, 1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output patch predictions
            nn.Conv2d(features*8, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)