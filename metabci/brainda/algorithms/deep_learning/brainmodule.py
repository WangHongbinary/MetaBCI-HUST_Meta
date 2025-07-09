import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch import Tensor
from math import fmod

def weights_init(m):
    """
    一个更健壮和符合常规的权重初始化函数。

    - 对卷积层 (Conv) 和全连接层 (Linear) 使用 Xavier 均匀分布初始化权重。
    - 对批量归一化层 (BatchNorm) 的权重初始化为1，偏置初始化为0。
    """
    # 获取模块的类名，例如 'Conv2d', 'Linear', 'BatchNorm2d'
    classname = m.__class__.__name__
    
    # 1. 初始化卷积层和全连接层
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # 使用 Xavier 均匀分布初始化权重
        # 这种方法有助于在网络层之间保持信号的方差，防止梯度消失或爆炸
        torch.nn.init.xavier_uniform_(m.weight)
        # 如果存在偏置项，则将其初始化为0
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
    # 2. 初始化批量归一化层
    elif classname.find('BatchNorm2d') != -1:
        # BatchNorm 的 'weight' (gamma) 通常初始化为 1
        # 'bias' (beta) 初始化为 0
        # 这样在训练开始时，BatchNorm 层不会改变其输入的均值和方差
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
          
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
                 out_channels=120,
                 num_classes=4):
        super().__init__()


        self.in_channels = data_dim #+ subject_dim
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

        self.classifier_head = nn.Sequential(
            OrderedDict(
                [
                    # 1. 全局平均池化，将 [B, C, H, W] -> [B, C, 1, 1]
                    # 在这里，它将 [32, 16, 1, 2100] -> [32, 16, 1, 1]
                    ('avg_pool', nn.AdaptiveAvgPool2d((1, 1))),
                    
                    # 2. 展平，将 [B, C, 1, 1] -> [B, C]
                    # 在这里，它将 [32, 16, 1, 1] -> [32, 16]
                    ('flatten', nn.Flatten()),
                    
                    # 3. 全连接层，进行分类
                    # 在这里，它将 [32, 16] -> [32, 4]
                    ('fc', nn.Linear(in_features=self.out_channels, out_features=num_classes))
                ]
            )
        )
        # 4. 在初始化结束时重置参数
        self.apply(weights_init)

    def forward(self, data: Tensor, subject_id: Tensor) -> Tensor:
        """简洁的 forward 方法"""
        
        # batch, _, _, length = data.size()
        # subject_vec = self.subject_embedding(subject_id)
        # subject_vec_expanded = subject_vec.view(batch, -1, 1, 1).expand(-1, -1, -1, length)
        
        # print("subject_vec.shape", subject_vec.shape, "subject_vec_expanded.shape",subject_vec_expanded.shape, "data.shape:", data.shape)
        # x = torch.cat([data, subject_vec_expanded], dim=1)
        
        # print("x.shape",x.shape)
        # --- 模型执行 ---
        # 所有操作都封装在 self.model 中，调用非常简洁
        
        data = data.unsqueeze(dim=2)
        features = self.model(data) 
        
        # --- 将特征送入分类头 ---
        output = self.classifier_head(features) # 输出维度: [32, 4]
        return output

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

    model = BrainModule(n_subjects=1, 
                data_dim=20, 
                subject_dim=16, 
                out_channels=16,
                num_classes=4)
    input = torch.randn(32, 20, 2100)
    subject = torch.tensor([i for i in range(1)])
    out = model(input, subject)
    print("input shape:", input.shape)
    print('output shape:', out.shape)

