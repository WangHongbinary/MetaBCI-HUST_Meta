"""
tonal_BCI_decoding network.
Modified from https://zenodo.org/records/13893640

"""

from collections import OrderedDict
import torch.nn as nn
import torch

class tonal_Net1(nn.Module):
    def __init__(self, *, typeNum, in_chans, num_layers=4, gruDim=256, drop_out=0.5):
        super().__init__()

        elec_feature = int(2*gruDim)
        time_kernel = 3
        pool_kernel = 2
        negative_slope = 0.01

        # time convolution
        self.step1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "time_conv",
                        nn.Conv1d(
                            in_chans,
                            gruDim,
                            time_kernel,
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    ("leaky_relu", nn.LeakyReLU(negative_slope)),
                    ("max_pool", nn.MaxPool1d(kernel_size=pool_kernel, stride=None, padding=0))
                ]
            )
        )

        gru_layers = []
        for i in range(num_layers):
            if i == 0:
                gru_layers.append(nn.GRU(gruDim, 
                                         gruDim, 
                                         1, 
                                         batch_first=True, 
                                         bidirectional=True))
            else:
                gru_layers.append(nn.GRU(gruDim * 2, 
                                         gruDim, 
                                         1, 
                                         batch_first=True, 
                                         bidirectional=True))
        self.gru_layers = nn.ModuleList(gru_layers)

        self.drop = nn.Dropout(p=drop_out)
        self.fc_layer = nn.Linear(elec_feature, typeNum)

    def forward(self, x):
        x = self.step1(x)
        x = x.permute(0, 2, 1)

        for gru_layer in self.gru_layers:
            x, _ = gru_layer(x)
            x = self.drop(x)
        
        x = x[:, -1, :]
        x = self.fc_layer(x)

        return x
    

class tonal_Net2(nn.Module):
    def __init__(self,
                *, 
                typeNum, 
                in_chans, 
                is_timespat,
                 n_filters_time,
                 filter_time_length,
                 n_filters_spat,
                 conv_stride,
                 pool_time_length,
                 pool_stride,
                 n_filters,
                 filter_length, 
                 n_CNN_layer,
                 gruDim,
                 gruLayer,
                 drop_out):
        super().__init__()

        self.is_timespat = is_timespat
        
        # time+spat convolution | time_spat convolution
        if self.is_timespat:
            self.step1 = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "time_spat_conv",
                            nn.Conv2d(
                                1,
                                n_filters_spat,
                                (filter_time_length, in_chans),
                                stride=(conv_stride, 1),
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(n_filters_spat, affine=True, eps=1e-5)),
                        ("elu", nn.ELU()),
                        ("pool", nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(pool_stride, 1))),
                    ]
                )
            )
        else:
            self.step1 = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "time_conv",
                            nn.Conv2d(
                                1,
                                n_filters_time,
                                (filter_time_length, 1),
                                stride=1,
                            ),
                        ),
                        (
                            "spat_conv",
                            nn.Conv2d(
                                n_filters_time,
                                n_filters_spat,
                                (1, in_chans),
                                stride=(conv_stride, 1),
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(n_filters_spat, affine=True, eps=1e-5)),
                        ("elu", nn.ELU()),
                        ("pool", nn.MaxPool2d(kernel_size=(pool_time_length, 1), stride=(pool_stride, 1))),
                    ]
                )
            )
        
        # conv_pool_block
        self.step2 = nn.ModuleList()
        self.step2.append(nn.Dropout(p=drop_out))
        self.step2.append(nn.Conv2d(
            n_filters_spat,
            n_filters,
            (filter_length, 1),
            stride=(conv_stride, 1),
            padding=(((filter_length - 1) * conv_stride) // 2,0)
        ))
        
        for i in range(n_CNN_layer-1):
            self.step2.append(nn.Dropout(p=drop_out))
            self.step2.append(nn.Conv2d(
                n_filters,
                n_filters,
                (filter_length, 1),
                stride=(conv_stride, 1),
                padding=(((filter_length - 1) * conv_stride) // 2,0)
            ))
            self.step2.append(nn.BatchNorm2d(
                n_filters,
                momentum=0.1,
                affine=True,
                eps=1e-5,
            ))
            self.step2.append(nn.ELU())
            self.step2.append(nn.MaxPool2d(
                kernel_size=(pool_time_length, 1),
                stride=(pool_stride, 1),
            ))

        self.gru_layer = nn.GRU(n_filters, gruDim, gruLayer, batch_first=True, bidirectional=True)
        elec_feature = int(2*gruDim)
        self.fc_layer = nn.Linear(elec_feature, typeNum)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.step1(x)
        
        for block in self.step2:
            x = block(x)

        x = x.squeeze()
        x = x.permute(0, 2, 1)

        x = self.gru_layer(x)[0][:,-1,:]
        x = self.fc_layer(x)
        
        return x
    
if __name__ == "__main__":
    net = tonal_Net1(typeNum=4, in_chans=20)
    input = torch.randn(32, 20, 2100)
    out = net(input)
    print(out.shape)

    
    filter_time_length = 2
    n_filters = 64
    conv_stride=3
    pool_time_length=3
    pool_stride=3
    filter_length=2
    n_CNN_layer=1
    gruDim=128
    gruLayer=1
    drop_out=0.8


    net2 = tonal_Net2(typeNum=4, 
                      in_chans=20,
                      is_timespat=True,
                      n_filters_time=n_filters, 
                      filter_time_length=filter_time_length,
                      n_filters_spat=n_filters, 
                      conv_stride=conv_stride,
                      pool_time_length=pool_time_length, 
                      pool_stride=pool_stride,
                      n_filters=n_filters, 
                      filter_length=filter_length, 
                      n_CNN_layer=n_CNN_layer,
                      gruDim=gruDim,
                      gruLayer=gruLayer,
                      drop_out=drop_out)
    input = torch.randn(32, 20, 2100)
    out = net2(input)
    print('input.shape:', input.shape)
    print('out.shape:', out.shape)