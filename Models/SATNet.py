import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import numpy as np

class InceptionBlk(nn.Module):
    def __init__(self, inch, outch):
        super(InceptionBlk, self).__init__()
        self.b1_conv1 = nn.Conv1d(inch, outch, kernel_size=1)
        self.b1_lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.b2_conv1 = nn.Conv1d(inch, outch, kernel_size=1)
        self.b2_lrelu1 = nn.LeakyReLU(negative_slope=0.3)
        self.b2_conv2 = nn.Conv1d(outch, outch, kernel_size=5, padding=2)
        self.b2_lrelu2 = nn.LeakyReLU(negative_slope=0.3)
        
        self.b3_conv1 = nn.Conv1d(inch, outch, kernel_size=1)
        self.b3_lrelu1 = nn.LeakyReLU(negative_slope=0.3)
        self.b3_conv2 = nn.Conv1d(outch, outch, kernel_size=9, padding=4)
        self.b3_lrelu2 = nn.LeakyReLU(negative_slope=0.3)
        
        self.b4_mpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.b4_conv1 = nn.Conv1d(inch, outch, kernel_size=1)
    
    def forward(self, x):
        x1 = self.b1_conv1(x)
        x1 = self.b1_lrelu(x1)

        x2 = self.b2_conv1(x)
        x2 = self.b2_lrelu1(x2)
        x2 = self.b2_conv2(x2)
        x2 = self.b2_lrelu2(x2)

        x3 = self.b3_conv1(x)
        x3 = self.b3_lrelu1(x3)
        x3 = self.b3_conv2(x3)
        x3 = self.b3_lrelu2(x3)

        x4 = self.b4_mpool(x)
        x4 = self.b4_conv1(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x

    
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

class TransformerTimeSeries(nn.Module):
    """
    基于 Transformer 的时序预测模型。
    模型结构包括：输入嵌入层 -> Transformer 模块（Encoder-Decoder 结构） -> 全连接输出层。
    
    参数：
    input_dim  : 输入特征维度（本例为1）
    model_dim  : Transformer 模型的隐藏层维度
    num_heads  : 注意力头数
    num_layers : Transformer 层数（Encoder 和 Decoder 均采用相同层数）
    dropout    : dropout 概率，用于防止过拟合
    """
    def __init__(self, input_dim=1, model_dim=32, num_heads=4, num_layers=2, dropout=0.1, output_channel=64):
        super(TransformerTimeSeries, self).__init__()
        # 输入嵌入层，将一维输入映射到 model_dim 维度
        self.embedding = nn.Linear(input_dim, model_dim)
        # 构造 Transformer 模块，采用 nn.Transformer 内置实现，Encoder 和 Decoder 均采用相同输入（自编码器方式）
        self.transformer = nn.Transformer(d_model=model_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dropout=dropout,batch_first=True)
        # 输出全连接层，将 Transformer 输出映射回一维预测值
        self.fc_out = nn.Linear(model_dim, output_channel)
        
    def forward(self, x):
        """
        前向传播函数。
        参数：
        x: 输入张量，形状为 (batch_size, seq_length, input_dim)
        
        返回：
        输出张量，形状为 (batch_size, seq_length, 1)
        """
        # 将输入数据经过嵌入层映射到高维空间
        x = self.embedding(x)
        # Transformer 模型要求的输入形状为 (seq_length, batch_size, model_dim)，因此进行转置
        # x = x.permute(1, 0, 2)
        # 这里使用自注意力机制，Encoder 和 Decoder 均输入相同数据
        x = self.transformer(x, x)
        # 将 Transformer 输出映射为目标维度
        x = self.fc_out(x)
        # 恢复原始形状 (batch_size, seq_length, 1)
        # x = x.permute(1, 0, 2)
        return x

class SATNet(nn.Module):
    def __init__(self, PreNum=3, PreLen=1, cluster_num=6):
        super(SATNet, self).__init__()
        self.scale = 48
        firstFilter = 8
        inceptionFilter = cluster_num * firstFilter
        self.FirConv = FirstConvBlk(outch=firstFilter, cluster_num=cluster_num)
        self.InceptionBlk1 = InceptionBlk(inch=inceptionFilter, outch=int(self.scale/4))
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.3)
        self.transformer = TransformerTimeSeries(inceptionFilter, model_dim=64, dropout=0, output_channel=PreNum*PreLen)

        

    def forward(self, x):
        x = self.FirConv(x)
        x = self.InceptionBlk1(x)
        x = self.lrelu1(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x[:, -1, :]
        return x
    
if __name__ == '__main__':
    model = SATNet(PreNum=3)
    model = model.cuda()
    inputs = torch.randn(24, 6, 256)
    inputs = inputs.cuda()
    outputs = model(inputs)
    print("outpus's shape :", outputs.shape)
