import torch 
import torch.nn as nn
import torch.nn.functional as F
 
class AECNN(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.encoder = AECNNEncoder(kernel_size = kernel_size)
        self.decoder = AECNNDecoder(kernel_size = kernel_size)
        
    def forward(self, x):
        x, skip = self.encoder(x)
        output = self.decoder(x, skip)
    
        return output
    
class AECNNEncoder(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

        self.out_channels = [2**(6+idx//3) for idx in range(8)] #[64, 64, 64, 128, 128, 128, 256, 256]
        self.encoder_layers = nn.ModuleList(
            [(nn.Conv1d(
                in_channels = 1 if idx ==0 else self.out_channels[idx-1],
                out_channels = self.out_channels[idx],
                kernel_size = self.kernel_size,
                stride = 1 if idx == 0 else 2,
                padding = (self.kernel_size-1)// 2
                ))for idx in range(8)])
        self.prelu_layers = nn.ModuleList(
            [(
                nn.PReLU()
            )for _ in range(8)]
        )
        
        self.bottleneck = nn.Conv1d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = self.kernel_size,
            stride = 2,
            padding = (self.kernel_size - 1) // 2
        )

    def forward(self,x):
        skip = []
        for idx in range(8):
            x =self.encoder_layers[idx](x)
            x = self.prelu_layers[idx](x)
            if idx%3 ==2:
                x = F.dropout(x, p = 0.2)
            skip.append(x)
            

        x = self.bottleneck(x)
        x = F.dropout(x, p = 0.2)

        return x, skip
    
class AECNNDecoder(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        out_channels = [2**(6+idx//3) for idx in range(8)]
        self.out_channels = out_channels[::-1]
        
        self.decoder_layers = nn.ModuleList(
            [(nn.ConvTranspose1d(
                in_channels = 256 if idx ==0 else self.out_channels[idx-1]*2,
                out_channels = self.out_channels[idx],
                kernel_size = self.kernel_size,
                stride = 2,
                padding = (self.kernel_size-1)// 2,
                output_padding = 1
                ))for idx in range(8)])
        
        self.prelu_layers = nn.ModuleList(
            [(
                nn.PReLU()
            )for _ in range(8)]
        )

        self.output_layer = nn.Conv1d(
            in_channels = 128,
            out_channels = 1,
            kernel_size = self.kernel_size,
            stride = 1,
            padding = (self.kernel_size - 1) // 2
        )

    def forward(self, x, skip):
        skip = skip[::-1]
        
        for idx in range(8):
            x = x if idx == 0 else torch.cat([x, skip[idx-1]], dim = 1)
            x = self.decoder_layers[idx](x)
            x = self.prelu_layers[idx](x)
            if idx%3 ==2:
                x = F.dropout(x, p = 0.2)
        
        x = torch.cat([x, skip[7]] ,dim = 1)
        x = self.output_layer(x)
        output = F.tanh(x)
    
        return output