"""
A New Framework for CNN-Based Speech Enhancement in the Time Domain
"""
 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import *

class AECNN(nn.Module):
    def __init__(self, in_channels=1, out_channels = 1, num_layers = NUM_LAYERS, kernel_size=KERNEL_SIZE):
        super().__init__()
 
        self.name= "AECNN"
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_layers = num_layers #encoder과 decoder의 layer number
        self.kernel_size= kernel_size

        self.Encoder = AECNN_Encoder(in_channels, num_layers, kernel_size)
        self.Decoder = AECNN_Decoder(out_channels, num_layers, kernel_size)
     
    def forward(self, x):
        '''
        while len(x.size()) < 2 :
            x= x.unsqueeze(-2)
        '''

        x_len= x.shape[-1]

        x, down = self.Encoder(x)

        x_enh = self.Decoder(x, down)[..., :x_len]

        #print(x_enh.shape)g
        return x_enh

    def get_name(self):
        return self.name

class AECNN_Encoder(nn.Module):
    def __init__(self, in_channels = 1, num_layers = 8, kernel_size = 11):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.down_channels = [2**(6+idx//3) for idx in range(num_layers)]

        self.unet_down = nn.ModuleList([UNet_down(
            in_channels = self.down_channels[idx-1] if idx>0 else in_channels,
            out_channels =self.down_channels[idx],
            kernel_size = kernel_size,
            stride = 2 if idx > 0 else 1,
            dropout = 0.2 if idx%3 == 2 else 0,
            bias=True
        ) for idx in range(self.num_layers)])

        self.unet_bottle = UNet_down(
            in_channels= 2**(6+(num_layers -1)//3),
            out_channels= 2**(6+num_layers//3),
            kernel_size= kernel_size,
            bias=True,
            stride=2,
            dropout=0.2 if num_layers % 3 == 2 else 0,
        )

    def forward(self,x):
        '''
        while len(x.size())< 2:
            x=x.unsqueeze(-2)
        '''

        down = []
        for idx in range(self.num_layers):
            x=self.unet_down[idx](x)
            down.append(x)

        x= self.unet_bottle(x)
    
        return x, down

class AECNN_Decoder(nn.Module):
    def __init__(self, out_channels=1, num_layers= 8 , kernel_size = 11):
        super().__init__()

        self.out_channels = out_channels
        self.num_layers = num_layers
        down_channels = [2**(6+idx//3) for idx in range(self.num_layers)]
        up_channels = list(reversed(down_channels))

        self.unet_up = nn.ModuleList([UNet_up(
            in_channels = down_channels[-idx] + up_channels[idx-1] if idx>0 else down_channels[-1],
            out_channels = up_channels[idx],
            kernel_size = kernel_size,
            stride = 2, 
            activation = "prelu",
            dropout = 0.2 if idx%3 == 2 else 0,
            bias = True
        ) for idx in range(self.num_layers)])

        self.unet_final = UNet_up(
            in_channels = down_channels[0] + up_channels[-1],
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = 1,
            activation = None,
            dropout = 0,
            bias = True
        )

    def forward(self, x, down):

        for idx in range(self.num_layers):
            x = self.unet_up[idx](x, down[-idx-1])

        x = self.unet_final(x, None)
        return x


class UNet_down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0, bias= True):
        super().__init__()        
        self.conv = nn.Conv1d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride,
                              dilation = 1, bias=bias, padding =kernel_size//2)
        #nn.init.orthogonal_(self.conv.weight)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout = dropout
        
        self.activation = nn.LeakyReLU()
        
        if dropout>0:
            self.do = nn.Dropout(dropout)
        
        
    def forward(self, x):
        l = x.shape[-1]
        x = F.pad(x, pad=(0, self.kernel_size))
        x = self.conv(x)
        
        
        x = x[..., :l//self.stride+1]
        
        x = self.activation(x)

        if self.dropout:
            x = self.do(x)
               
        
        return x

class UNet_up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 2, activation = "leaky_relu", dropout = 0, bias = True, r = 2):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = kernel_size//2,
            bias = bias
        )
        

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.r = r
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU() if activation != None else activation

    def forward(self, x, x_prev):
        x = self.conv(x)
        if self.activation is not None: 
            x = self.activation(x)

        if self.dropout:
            x = self.dropout(x)

        if x_prev is not None:
            x = torch.cat([x[..., :x_prev.shape[-1]], x_prev], dim=1)
        
        

        return x



if __name__ == '__main__':
    model = AECNN()
    model.cuda()
    
    

    y = torch.randn(4, 1, 512).cuda()
    z = model(y)

    print(z.shape)
    
    del model
    