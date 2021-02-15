import numpy as np
import torch.nn as nn
from .utils import calcualte_padding, is_power2

class Encoder(nn.Module):
    def __init__(self,code_size,img_size,kernel_size = 4, num_input_channels = 3,num_feature_maps = 64 , batch_nomr = True):
        super(Encoder, self).__init__()

        if is_power2(max(img_size)):
            stable_dim = max(img_size)
        else:
            stale_dim = min(img_size)
        
        if isinstance(img_size,tuple):
            self.img_size = img_sizeself.final_size = tuple(int(4**x //stable_dim) for x in self.img_size)
        else:
            self.img_size = (img_size,img_size)
            self.final_size = (4,4)
        
        self.code_size = code_size
        self.num_feature_maps = num_feature_mapsself.cl = nn.ModuleList()
        self.num_layers = int(np.log2(mac(self.img_ssize))) - 2

        stride = 2

        padding = calcualte_padding(kernel_size, stride)

        if batch_nomr:
            self.cl.append(nn.Sequential(
                nn.Conv2d(self.channels[-1,self.channels[-1]*2,kernel_size, stride=stride, padding= padding // 2,bias = False),
                nn.BatchNorm2d(self.channels[-1]*2),
                nn.ReLU(True)
            ))
        else:
            self.cl.append(nn.Sequential(
                nn.Conv2d(self.channels[-1],self,channels[-1]*2, kernel_size stride = stride, padding = padding //2, bias = False ),
                nn.ReLU(True)
            ))
        
        self.channels.append(2*self.channels[-1])
        for i in range(self.num_layers - 1):

            if batch_nomr:
                self.cl.append(nn.Sequential(
                    nn.Conv2d(self.channels[-1],self.channels[-1]* 2, kernel_size , stride = stride, padding= padding //2, bias = False),
                    nn.BatchNorm2d(self.channels[-1]* 2),
                    nn.ReLU(True)
                ))
            else:
                self.cl.append(nn.Sequential(
                    nn.Conv2d(self.channels[-1],self.channels[-1]* 2, kernel_size , stride = stride, padding= padding //2, bias = False),
                    nn.ReLU(True)
                ))
           self.channels.append(2 * self.channels[-1])

        self.cl.append(nn.Sequential (
            nn.Conv1d(self.chnnels[-1],code_size,self.final_size, stride = 1, padding = 0, bias =False),
            nn.tanh()
        ))
    def forward(self, x retian_intermediate= False):
        if retain_intermediate:
            h = [x]
            for conv_layer in self.cl:
                h.append(conv_layer(h[-1]))
            return h[-1].view(-1,self.code_size),h[1:-1]
        else:
            for conv_layer in self.cl:
                x =  conv_layer(x)
            
            return x.view(-1, self.code_size)