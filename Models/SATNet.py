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
        def __init__(self, outch, input_shape = [256, 6]):
            
            super(FirstConvBlk, self).__init__()        
            # Shallow feature extraction module
            self.first_conv_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, outch, kernel_size=9, padding=4),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.MaxPool1d(4),
                    # nn.Dropout(0.2)
                ) for _ in range(input_shape[1])
                ])
        def forward(self, x):
            first_conv_outs = []
            for i in range(x.shape[1]):
                x_input = x[:, i:i+1, :]  # Slice operation
                out = self.first_conv_blocks[i](x_input)
                first_conv_outs.append(out)
        
            x = torch.cat(first_conv_outs, dim=1)
            return x
        

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=48, d_model=64, nhead=4, num_layers=2, dropout=0.1, pre_num=3):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)  # 输入特征映射
        self.pos_encoder = PositionalEncoding(d_model, dropout)  # 位置编码
        # Transformer编码器层（含因果掩码，避免未来信息泄露）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True,
            norm_first=True  # 先归一化，再残差连接，提升稳定性
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, pre_num)  # 输出层
        
    def forward(self, x):
        # 输入形状：(batch_size, seq_length, input_dim)
        x = self.embedding(x) * np.sqrt(self.d_model)  # 缩放嵌入，稳定梯度
        x = self.pos_encoder(x)  # 注入位置信息
        # 因果掩码：确保第i步只能看到前i步的信息
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        x = self.transformer_encoder(x, src_key_padding_mask=None, mask=mask)
        output = self.fc(x)  # 输出形状：(batch_size, seq_length, 1)
        return output
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        # 预计算位置编码，避免重复计算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x形状：(batch_size, seq_length, d_model)
        x = x + self.pe[:x.size(1), :]  # 叠加位置编码
        # x = self.dropout(x)
        return x

class SATNet(nn.Module):
    def __init__(self, PreLen=1, PreNum=1, input_shape = [256, 6]):
        super(SATNet, self).__init__()
        firstFilter = 8
        inceptionFilter = input_shape[1] * firstFilter

        self.FirConv = FirstConvBlk(outch=firstFilter, input_shape=input_shape)
        self.InceptionBlk1 = InceptionBlk(inch=inceptionFilter, outch=int(inceptionFilter/4))
        self.InceptionBlk2 = InceptionBlk(inch=inceptionFilter, outch=inceptionFilter)
        self.tranformer = TransformerPredictor(inceptionFilter*4, 64*4, 4, 2, 0.1, pre_num=3)
        

    def forward(self, x):
        x = self.FirConv(x)
        x = self.InceptionBlk(x)
        x = x.permute(0, 2, 1)
        x = self.tranformer(x)

        return x
    
if __name__ == '__main__':
    model = SATNet()
    model = model.cuda()
    inputs = torch.randn(24, 6, 256)
    inputs = inputs.cuda()
    outputs = model(inputs)
    print("outpus's shape :", outputs.shape)
