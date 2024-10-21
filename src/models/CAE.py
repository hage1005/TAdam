import torch
import torch.nn as nn
import torch.nn.functional as F

class CAE(nn.Module):
    def __init__(self, in_shape, filters, code_dim):
        super(CAE, self).__init__()
        self.in_shape = in_shape
        self.filters = filters
        self.code_dim = code_dim
        
        # Encoder
        self.enc01 = nn.Conv2d(in_channels=in_shape[0], out_channels=filters[0], kernel_size=3, stride=2, padding=1)
        self.enc02 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, stride=2, padding=1)
        self.enc03 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.code_layer = nn.Linear(filters[2] * (in_shape[1] // 6) * (in_shape[2] // 6), code_dim)

        # Decoder
        self.dec03_fc = nn.Linear(code_dim, filters[2] * (in_shape[1] // 6) * (in_shape[2] // 6))
        self.dec03_reshape = nn.Unflatten(1, (filters[2], in_shape[1] // 6, in_shape[2] // 6))
        self.dec02 = nn.ConvTranspose2d(in_channels=filters[2], out_channels=filters[1], kernel_size=3, stride=2, padding=1)
        self.dec01 = nn.ConvTranspose2d(in_channels=filters[1], out_channels=filters[0], kernel_size=3, stride=2, padding=1, output_padding=1)

        # Output layer
        self.output_layer = nn.ConvTranspose2d(in_channels=filters[0], out_channels=in_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder
        x = F.silu(self.enc01(x))  # Using SiLU (Swish activation)
        x = F.silu(self.enc02(x))
        x = F.silu(self.enc03(x))
        
        # Flatten and code
        x = self.flatten(x)
        code = self.code_layer(x)
        
        # Decoder
        x = F.silu(self.dec03_fc(code))
        x = self.dec03_reshape(x)
        x = F.silu(self.dec02(x))
        x = F.silu(self.dec01(x))
        
        # Output
        x = self.output_layer(x)
        return x
