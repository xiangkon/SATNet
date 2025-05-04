import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

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
    
# class NNWA(nn.Module):
#     def __init__(self, IndexL):
#         super(NNWA, self).__init__()
#         self.LL = nn.ModuleList()  # 使用 ModuleList 来管理层
#         for i in range(len(IndexL)):
#             if len(IndexL[i]) != 1:
#                 self.LL.append(nn.LazyLinear(1))  

#         modified_list = []
#         for row in IndexL:
#             new_row = [value - 1 for value in row]
#             modified_list.append(new_row)

#         self.IndexL = modified_list

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         device = x.device  # 获取输入所在的设备
#         xL = []
#         for i in range(len(self.IndexL)):
#             xL.append(x[:, :, self.IndexL[i]].to(device))

#         xList = []
#         x = xL
#         k = 0
#         for i in range(len(self.IndexL)):
#             if len(self.IndexL[i]) != 1:
#                 xList.append(self.LL[k](x[i]))
#                 k += 1
#             else:
#                 xList.append(x[i])
            
#         x = torch.cat(xList, dim=2)
#         return x.permute(0,2,1)


class NNWA(nn.Module):
    def __init__(self, IndexL):
        super(NNWA, self).__init__()  
        self.l1 = nn.Linear(len(IndexL[0]), 1, bias=False)
        self.l2 = nn.Linear(len(IndexL[1]), 1, bias=False)
        self.l3 = nn.Linear(len(IndexL[2]), 1, bias=False)
        self.l4 = nn.Linear(len(IndexL[3]), 1, bias=False)
        self.l5 = nn.Linear(len(IndexL[4]), 1, bias=False)
        self.l6 = nn.Linear(len(IndexL[5]), 1, bias=False)
        torch.nn.init.constant_(self.l1.weight, 1/len(IndexL[0]))
        torch.nn.init.constant_(self.l2.weight, 1/len(IndexL[1]))
        torch.nn.init.constant_(self.l3.weight, 1/len(IndexL[2]))
        torch.nn.init.constant_(self.l4.weight, 1/len(IndexL[3]))
        torch.nn.init.constant_(self.l5.weight, 1/len(IndexL[4]))
        torch.nn.init.constant_(self.l6.weight, 1/len(IndexL[5]))
        modified_list = []
        for row in IndexL:
            new_row = [value - 1 for value in row]
            modified_list.append(new_row)

        self.IndexL = modified_list

    def forward(self,x):
        x = x.permute(0,2,1)
        x1 = x[:, :, self.IndexL[0]]
        x2 = x[:, :, self.IndexL[1]]
        x3 = x[:, :, self.IndexL[2]]
        x4 = x[:, :, self.IndexL[3]]
        x5 = x[:, :, self.IndexL[4]]
        x6 = x[:, :, self.IndexL[5]]

        x1 = self.l1(x1)
        x2 = self.l2(x2)
        x3 = self.l3(x3)
        x4 = self.l4(x4)
        x5 = self.l5(x5)
        x6 = self.l6(x6)

        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=2)
        return x.permute(0,2,1)

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


class SeEANet_N(nn.Module):
    def __init__(self, PreNum=3, PreLen=1, cluster_num=6, IndexL=[[11, 12], [9, 10], [3], [4], [6, 8], [1, 2, 5, 7]]):
        super(SeEANet_N, self).__init__()
        firstFilter = 8
        inceptionFilter = cluster_num * firstFilter
        self.nnwa = NNWA(IndexL)
        self.FirConv = FirstConvBlk(outch=firstFilter, cluster_num=cluster_num)
        self.InceptionBlk1 = InceptionBlk(inch=inceptionFilter, outch=inceptionFilter)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.3)
        
        self.mpool1 = nn.MaxPool1d(kernel_size=3)
        self.conv1 = nn.LazyConv1d(out_channels=64, kernel_size=1)

        self.gru1 = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        # self.l0 = nn.LazyLinear(64)
        # self.relu1 = nn.ReLU()
        self.l1 = nn.LazyLinear(PreLen*PreNum)
        

    def forward(self, x):
        x = self.nnwa(x)
        x = self.FirConv(x)
        x = self.InceptionBlk1(x)
        x = self.lrelu1(x)
        x = self.mpool1(x)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x,_ = self.gru1(x)
        x,_ = self.gru2(x)
        x = x[:, -1, :]
        # x = self.l0(x)
        # x = self.relu1(x)
        x = self.l1(x)


        return x
    
if __name__ == '__main__':
    # model = SeEANet_N(PreNum=3)
    model = NNWA([[11, 12], [9, 10], [3], [4], [6, 8], [1, 2, 5, 7]])
    print(model.l1.weight)
    print(model.l2.weight)
    print(model.l3.weight)
    print(model.l4.weight)
    print(model.l5.weight)
    print(model.l6.weight)
    print("---------------")
    print(model.l1.bias)
    print(model.l2.bias)
    print(model.l3.bias)
    print(model.l4.bias)
    print(model.l5.bias)
    print(model.l6.bias)
    # model = model.cuda()
    # inputs = torch.randn(24, 12, 256)
    # inputs = inputs.cuda()
    # # summary(model, input_size=(6, 256))
    # outputs = model(inputs)
    # print("outpus's shape :", outputs.shape)
