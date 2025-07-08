import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch import Tensor
from math import fmod

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        # 💡 提示: BatchNorm2d 的权重通常初始化为1，偏置为0。
        # torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        # torch.nn.init.constant_(m.bias.data, 0.0)
        # 这里保留你的实现：
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
          
#定义空间注意力层（作用不明显，暂时用1*1卷积代替）
class Spatial_attention(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Spatial_attention, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1, 1), stride=1))

    def forward(self, x):
        out = self.block(x)
        return out

# 空间注意力模块
class SA_Layer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(SA_Layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            Spatial_attention(in_channels=in_channels, out_channels=270),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=270, out_channels=270, kernel_size=(1, 1), stride=1))
    
    def forward(self, x):
        out = self.block(x)
        return out

# Subject Layer
class Subject_Layer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Subject_Layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), stride=1)

    def forward(self, x):
        out = self.layer(x)
        return out

# Residual block
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, k) -> None:
        super(Residual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k # number of layer k=1,2,3,4
        self.p_d1 = int(pow(2, fmod(2 * self.k, 5))) # Residual_layer_1，padding = dilation = self.p_d1
        self.p_d2 = int(pow(2, fmod(2 * self.k + 1, 5))) # Residual_layer_2，padding = dilation = self.p_d2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=320, kernel_size=(1, 3), stride=1, 
                      padding=(0, self.p_d1), dilation=(1, self.p_d1)),
            nn.BatchNorm2d(320),
            nn.GELU())
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=self.out_channels, kernel_size=(1, 3), stride=1, 
                      padding=(0, self.p_d2), dilation=(1, self.p_d2)),
            nn.BatchNorm2d(320))

        self.after_conv = nn.Sequential(
            nn.GELU())

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.after_conv(x + out)
        return out

class BrainModule(nn.Module):
    def __init__(self, 
                 n_subjects=15, 
                 data_dim=208, 
                 subject_dim=16,
                 out_channels=120):
        super().__init__()


        self.in_channels = data_dim + subject_dim
        self.out_channels = out_channels
        
        # 定义各层的通道数
        ch_sa = 270     #空间注意力
        ch_subject = 270    #subject_layer    
        ch_conv1_out = 320
        ch_res_in_out = 320
        ch_glu_in = 320
        ch_glu_out = 640
        ch_output_in = 320
        ch_output_mid = 640
        
        conv_kernel = [(1, 3), (1, 1)]
        conv_padding = [(0, 1), (0, 2)]
        conv_dilation = [1, (1, 2)]
        stride = [1]


        self.subject_embedding = nn.Embedding(n_subjects, subject_dim)

        
        # SA_Layer and Subject_Layer
        self.step1 = nn.Sequential(
            OrderedDict(
                [
                    ('sa_layer', SA_Layer(in_channels=self.in_channels, out_channels=ch_sa)),
                    ('subject_layer', Subject_Layer(in_channels=ch_sa, out_channels=ch_subject))
                ]
            )
        )
        
        # conv_block_1
        self.step2 = nn.Sequential(
            OrderedDict(
                [
                    ('conv1', nn.Conv2d(in_channels=ch_subject, out_channels=ch_conv1_out, kernel_size=conv_kernel[0], stride=stride[0], padding=conv_padding[0], dilation=conv_dilation[0])),
                    ('bn1', nn.BatchNorm2d(ch_conv1_out)),
                    ('gelu1', nn.GELU()),
                    ('conv2', nn.Conv2d(in_channels=ch_conv1_out, out_channels=ch_conv1_out, kernel_size=conv_kernel[0], stride=stride[0], padding=conv_padding[1], dilation=conv_dilation[1])),
                    ('bn2', nn.BatchNorm2d(ch_conv1_out)),
                    ('gelu2', nn.GELU()),
                    ('conv3', nn.Conv2d(in_channels=ch_conv1_out, out_channels=ch_glu_out, kernel_size=conv_kernel[0], stride=stride[0], padding=conv_padding[0], dilation=conv_dilation[0])),
                    ('glu', nn.GLU(dim=1))
                ]
            )
        )
        
        # conv_block_2
        self.step3 = nn.Sequential(
            OrderedDict(
                [
                    ('residual', Residual(in_channels=ch_res_in_out, out_channels=ch_res_in_out, k=1)), 
                    ('conv', nn.Conv2d(in_channels=ch_glu_in, out_channels=ch_glu_out, kernel_size=conv_kernel[0], stride=stride[0], padding=conv_padding[0], dilation=conv_dilation[0])),
                    ('glu', nn.GLU(dim=1))
                ]
            )
        )
        
        # conv_block_3
        self.step4 = nn.Sequential(
            OrderedDict(
                [
                    ('residual', Residual(in_channels=ch_res_in_out, out_channels=ch_res_in_out, k=2)), 
                    ('conv', nn.Conv2d(in_channels=ch_glu_in, out_channels=ch_glu_out, kernel_size=conv_kernel[0], stride=stride[0], padding=conv_padding[0], dilation=conv_dilation[0])),
                    ('glu', nn.GLU(dim=1))
                ]
            )
        )
        
        # conv_block_4
        self.step5 = nn.Sequential(
            OrderedDict(
                [
                    ('residual', Residual(in_channels=ch_res_in_out, out_channels=ch_res_in_out, k=3)), 
                    ('conv', nn.Conv2d(in_channels=ch_glu_in, out_channels=ch_glu_out, kernel_size=conv_kernel[0], stride=stride[0], padding=conv_padding[0], dilation=conv_dilation[0])),
                    ('glu', nn.GLU(dim=1))
                ]
            )
        )
        
        # conv_block_5
        self.step6 = nn.Sequential(
            OrderedDict(
                [
                    ('residual', Residual(in_channels=ch_res_in_out, out_channels=ch_res_in_out, k=4)), 
                    ('conv', nn.Conv2d(in_channels=ch_glu_in, out_channels=ch_glu_out, kernel_size=conv_kernel[0], stride=stride[0], padding=conv_padding[0], dilation=conv_dilation[0])),
                    ('glu', nn.GLU(dim=1))
                ]
            )
        )
        
        # output_block
        self.step7 = nn.Sequential(
            OrderedDict(
                [
                    ('conv1', nn.Conv2d(in_channels=ch_output_in, out_channels=ch_output_mid, kernel_size=conv_kernel[1], stride=stride[0])),
                    ('gelu', nn.GELU()),
                    ('conv2', nn.Conv2d(in_channels=ch_output_mid, out_channels=self.out_channels, kernel_size=conv_kernel[1], stride=stride[0]))
                ]
            )
        )
        
        self.model = nn.Sequential(self.step1, self.step2, self.step3, self.step4, self.step5, self.step6, self.step7)

        # 4. 在初始化结束时重置参数
        self.apply(weights_init)

    def forward(self, data: Tensor, subject_id: Tensor) -> Tensor:
        """简洁的 forward 方法"""
        
        batch, _, _, length = data.size()
        subject_vec = self.subject_embedding(subject_id)
        subject_vec_expanded = subject_vec.view(batch, -1, 1, 1).expand(-1, -1, -1, length)
        x = torch.cat([data, subject_vec_expanded], dim=1)
        
        # --- 模型执行 ---
        # 所有操作都封装在 self.model 中，调用非常简洁
        out = self.model(x)
        return out

    def cal_backbone(self, data: Tensor, subject_id: Tensor, **kwargs) -> Tensor:
        """提取主干网络特征的辅助方法"""
        # --- 数据准备 ---
        batch, _, _, length = data.size()
        subject_vec = self.subject_embedding(subject_id)
        subject_vec_expanded = subject_vec.view(batch, -1, 1, 1).expand(-1, -1, -1, length)
        x = torch.cat([data, subject_vec_expanded], dim=1)

        # --- 特征提取 ---
        # 遍历 self.model 中的所有步骤，但不包括最后一个（step7, 即 output_block）
        tmp = x
        for step in self.model[:-1]:
            tmp = step(tmp)
        return tmp

if __name__ == '__main__':

    model = BrainModule(n_subjects=2, 
                data_dim=208, 
                subject_dim=16, 
                out_channels=120)
    input = torch.randn(2, 208, 1, 360)
    subject = torch.tensor([0, 1])
    out = model(input, subject)
    print('output shape:', out.shape)
