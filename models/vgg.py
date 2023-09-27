"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512          ],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512          ],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512     ],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100, num_channels=3):
        super().__init__()
        self.features = features

        if num_channels == 3:
            dim = 4096
        else:
            dim = 256
        self.projector = nn.Sequential(
            nn.Linear(512, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.classifier = nn.Linear(dim, num_class)

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.classifier(z)

        result = {'output': z}
        if logit:
            result['logit'] = z
        return result

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:  #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.projector(output)
        result = {'representation' : output}
        output = self.classifier(output)
        result['output'] = output
        return result

def make_layers(cfg, batch_norm=False, num_channels=3):
    layers = []

    input_channel = num_channels
    if num_channels == 3:
        cfg.append('M')
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn(num_classes, num_channels=3):
    return VGG(make_layers(cfg['D'], batch_norm=True, num_channels=num_channels), num_class=num_classes, num_channels=num_channels)

def vgg19_bn(num_classes, num_channels=3):
    return VGG(make_layers(cfg['E'], batch_norm=True, num_channels=num_channels), num_class=num_classes)


if __name__ == '__main__':
    net = vgg16_bn(num_classes=10, num_channels=1)
    x = torch.randn(2,1,28,28)
    print(net(x)['output'].size())