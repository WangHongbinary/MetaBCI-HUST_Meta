import torch
from torch import nn
from collections import OrderedDict
from torch import Tensor
from torchsummary import summary

# 1. 创建一个统一的权重初始化函数
def weights_init(m: nn.Module):
    """
    统一初始化模型中的卷积层、批归一化层和线性层。
    """
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif classname.find('Linear') != -1:
        nn.init.trunc_normal_(m.weight, mean=0., std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, val=0.)

# 2. 将所有模块整合进一个类中
class CNN_GRU_Refactored(nn.Module):
    def __init__(self,
                 # Tokenizer 参数
                 in_channels=10, 
                 tokenizer_out_channels=128,
                 kernel_size=5,
                 stride=3,
                 padding=2,
                 # Encoder (GRU) 参数
                 gru_hidden_size=128,
                 gru_num_layers=2,
                 gru_dropout=0.2,
                 # Classifier 参数
                 n_classes=61,
                 cls_dropout=0.0
                 ):
        super().__init__()
        
        # --- 定义模型的各个步骤 ---
        
        # Step 1: 卷积分词器 (Convolutional Tokenizer)
        self.step1_tokenizer = nn.Sequential(OrderedDict([
            ('conv', nn.Conv1d(
                in_channels=in_channels, 
                out_channels=tokenizer_out_channels, 
                kernel_size=kernel_size,
                stride=stride, 
                padding=padding
            )),
            ('bn', nn.BatchNorm1d(num_features=tokenizer_out_channels))
        ]))

        # Step 2: GRU 编码器 (GRU Encoder)
        # 💡 GRU层有复杂的内部权重，通常Pytorch的默认初始化效果很好，
        #    因此我们不在 weights_init 中为它设置特殊的初始化。
        self.step2_encoder = nn.GRU(
            input_size=tokenizer_out_channels, 
            hidden_size=gru_hidden_size, 
            num_layers=gru_num_layers,
            batch_first=True, # 确保(batch, seq, feature)格式
            dropout=gru_dropout, 
            bidirectional=True # 双向
        )
        
        # Step 3: 分类头 (Classification Head)
        # 因为GRU是双向的，所以输入特征维度是 2 * gru_hidden_size
        classifier_in_features = 2 * gru_hidden_size
        self.step3_classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(classifier_in_features, gru_hidden_size)),
            ('dropout', nn.Dropout(cls_dropout)),
            ('fc2', nn.Linear(gru_hidden_size, n_classes)),
            ('activation', nn.Sigmoid())
        ]))
        
        # --- 应用统一的权重初始化 ---
        self.apply(weights_init)

    def forward(self, x: Tensor) -> Tensor:

        # x 初始形状: (batch, channels, length) e.g., (B, 10, L)
        
        # Step 1: Tokenizer
        # 输出形状: (B, 128, L_new)
        tokens = self.step1_tokenizer(x)
        
        # 为了输入GRU，需要将形状变为 (batch, seq_len, features)
        # 交换最后两个维度 -> (B, L_new, 128)
        tokens_permuted = tokens.permute(0, 2, 1)
        
        # Step 2: Encoder
        # gru_output 形状: (B, L_new, 2 * hidden_size)
        gru_output, _ = self.step2_encoder(tokens_permuted)
        
        # 仅取序列的最后一个时间步的输出用于分类
        # 形状: (B, 2 * hidden_size)
        last_time_step = gru_output[:, -1, :]
        
        # Step 3: Classifier
        # 输入形状: (B, 256) -> 输出形状: (B, n_classes)
        out = self.step3_classifier(last_time_step)
        
        return out

    def cal_backbone(self, x: Tensor) -> Tensor:
        """
        提取主干网络（Tokenizer + Encoder）的输出特征。
        """
        tokens = self.step1_tokenizer(x)
        tokens_permuted = tokens.permute(0, 2, 1)
        backbone_features, _ = self.step2_encoder(tokens_permuted)
        # 返回整个序列的特征，形状为 (B, L_new, 2 * hidden_size)
        return backbone_features

# --- 使用示例 ---
if __name__ == '__main__':
    # 创建模型实例
    model = CNN_GRU_Refactored(n_classes=61).cuda()
    
    # 打印模型结构
    summary(model, input_size=(10, 90))
    
    # 创建一个假的输入张量 (batch_size=4, channels=10, length=512)
    dummy_input = torch.randn(32, 10, 90).cuda()
    
    # 执行前向传播
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # 测试主干网络输出
    backbone_out = model.cal_backbone(dummy_input)
    print(f"Backbone output shape: {backbone_out.shape}")
    
    