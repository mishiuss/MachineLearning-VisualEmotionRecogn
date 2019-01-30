import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch.common.losses import *


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.xavier_normal_(m.weight.data, gain=np.sqrt(2))

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, data):
        x = data
        for name, module in self.submodule._modules.items():
            if len(module._modules.items()) != 0:
                for name2, module2 in module._modules.items():
                    x = module2(x)
            else:
                x = module(x)
        return x

class StatPoolLayer(nn.Module):
    # Building of the SGPK (Snyder, Garcia-Romero, Povey, Khudanpur) statistics pooling layer's model. Implementation
    # is based on work Snyder D. et al. X-vectors: Robust DNN embeddings for speaker recognition // ICASSP, Calgary,
    # Canada. – 2018
    def __init__(self):
        # Function for initializing of the class's instance
        super(StatPoolLayer, self).__init__()

    def forward(self, x):
        # Function for forward propagation
        # x - input (output) of the model
        dim = -1
        mean_x = x.mean(dim)
        mean_x2 = x.pow(2).mean(dim)
        std_x = F.relu(mean_x2 - mean_x.pow(2)).sqrt()
        return torch.cat([mean_x, std_x], dim=-1)

class STPNet(nn.Module):
    def __init__(self, num_classes, depth):
        super(STPNet, self).__init__()

        modules = nn.Sequential()
        modules.add_module('StatPoolLayer', StatPoolLayer())
        modules.add_module('fc1', nn.Linear(160, 512, bias=True))
        modules.add_module('pr1', nn.ReLU())
        modules.add_module('fc2', nn.Linear(512, 1024, bias=True))
        modules.add_module('pr2', nn.ReLU())
        modules.add_module('fc3', nn.Linear(1024, num_classes, bias=False))

        # init by xavier
        modules.apply(weights_init)
        self.net = FeatureExtractor(modules, [])

    def forward(self, data):
        output = self.net(data)
        return output

