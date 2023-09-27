#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

from models.resnetcifar import ResNet18_cifar10
from models.mobileNetV2 import MobileNetV2
from models.vgg import vgg16_bn
from models.Nets import CNNCifar, CNNMnist, ModelFedCon, CNNFashionMnist
from models.Update import *
from models.test import test_img
from models.Fed import Aggregation
from models.LSTM import CharLSTM