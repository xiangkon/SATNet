import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import numpy as np

class FirstConvBlk(nn.Module):
        def __init__(self, outch, cluster_num):
            
            super(FirstConvBlk, self).__init__()        
            # Shallow feature extraction module
            self.first_conv_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, outch, kernel_size=9, padding=4),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.MaxPool1d(4),
                    # nn.Dropout(0.2)
                ) for _ in range(cluster_num)
                ])
        def forward(self, x):
            first_conv_outs = []
            for i in range(x.shape[1]):
                x_input = x[:, i:i+1, :]  # Slice operation
                out = self.first_conv_blocks[i](x_input)
                first_conv_outs.append(out)
        
            x = torch.cat(first_conv_outs, dim=1)
            return x

class MyoNet(nn.Module):
    def __init__(self, PreLen=3, PreNum=1, cluster_num=6):
        super(MyoNet, self).__init__()
        firstFilter = 20
        self.conv1 = FirstConvBlk(outch=firstFilter, cluster_num=cluster_num)
        self.conv2 = nn.Conv1d(firstFilter*cluster_num, firstFilter, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool1d(kernel_size=4)
        self.lstm1 = nn.LSTM(input_size=firstFilter, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.l1 = nn.Linear(64, PreLen*PreNum)

    def forward(self,x):
         x = self.conv1(x)
         x = self.conv2(x)
         x = self.relu1(x)
         x = self.mp1(x)
         x = x.permute(0,2,1)
         x,_ = self.lstm1(x)
         x,_ = self.lstm2(x)
         x = x[:,-1,:]
         x = self.l1(x)
         return x

if __name__ == '__main__':
    model = MyoNet(PreNum=3)
    model = model.cuda()
    inputs = torch.randn(24, 6, 256)
    inputs = inputs.cuda()
    outputs = model(inputs)
    print("outpus's shape :", outputs.shape)