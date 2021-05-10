import torch
# from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, BCEWithLogitsLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, AdaptiveAvgPool2d
from torch.optim import Adam, SGD

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.models as models

class NeuralNetwork(Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.features = Sequential(
            Conv2d(in_channels =1, out_channels =8, kernel_size=5, stride=1, padding=1),
            BatchNorm2d(num_features=8),
            ReLU(inplace=True),
            MaxPool2d(2,2),

            Conv2d(in_channels =8, out_channels =16, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(num_features=16),
            ReLU(inplace=True),
            MaxPool2d(2,2),
            
            # Conv2d(in_channels =16, out_channels =32, kernel_size=3, stride=1, padding=0),
            # BatchNorm2d(num_features=32),
            # ReLU(inplace=True),
            # MaxPool2d(2,2),
            # Conv2d(in_channels =64, out_channels =128, kernel_size=3, stride=1, padding=0),
            # ReLU(inplace=True)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(8, 8))
        self.classifier = Sequential(
            Linear(in_features=16*8*8, out_features=2048, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            Linear(in_features=2048, out_features=1024, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            # Linear(in_features=1024, out_features=512, bias=True),
            # ReLU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            Linear(in_features=1024, out_features=1, bias=True)

        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # flattenning 
        x = x.view(-1,16*8*8)
        logits = self.classifier(x)
        return logits