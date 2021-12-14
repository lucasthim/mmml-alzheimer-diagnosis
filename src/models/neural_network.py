import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, AdaptiveAvgPool2d
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.features = Sequential(
            Conv2d(in_channels =1, out_channels =8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=8),
            ReLU(inplace=True),
            MaxPool2d(2,2),

            Conv2d(in_channels =8, out_channels =16, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(num_features=16),
            ReLU(inplace=True),
            MaxPool2d(2,2),
            
            Conv2d(in_channels =16, out_channels =32, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(num_features=32),
            ReLU(inplace=True),
            MaxPool2d(2,2),
            
            Conv2d(in_channels =32, out_channels =64, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(8, 8))
        self.classifier = Sequential(
            # Remember changing the x.view() number as well. It needs to be flattenend!
            Linear(in_features=64*8*8, out_features=512, bias=True),
            ReLU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            Linear(in_features=512, out_features=512, bias=True),
            ReLU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            # Linear(in_features=512, out_features=512, bias=True),
            # ReLU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            Linear(in_features=512, out_features=1, bias=True)

        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # flattenning 
        x = x.view(-1,64*8*8)
        logits = self.classifier(x)
        return logits

class SuperShallowCNN(Module):
    def __init__(self):
        super(SuperShallowCNN, self).__init__()
        self.features = Sequential(
            Conv2d(in_channels =1, out_channels =8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=8),
            ReLU(inplace=True),
            MaxPool2d(2,2),

            Conv2d(in_channels =8, out_channels =16, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(num_features=16),
            ReLU(inplace=True),
            MaxPool2d(2,2),
            
            Conv2d(in_channels =16, out_channels =32, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(num_features=32),
            ReLU(inplace=True),
            MaxPool2d(2,2),
            
            Conv2d(in_channels =32, out_channels =64, kernel_size=3, stride=1, padding=0),
            BatchNorm2d(num_features=64),
            ReLU(inplace=True),
            MaxPool2d(2,2),

            Conv2d(in_channels =64, out_channels =128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(4, 4))
        self.classifier = Sequential(
            # Remember changing the x.view() number as well. It needs to be flattenend!
            Linear(in_features=128*4*4, out_features=128, bias=True),
            ReLU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            Linear(in_features=128, out_features=64, bias=True),
            ReLU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            # Linear(in_features=64, out_features=64, bias=True),
            # ReLU(inplace=True),
            # Dropout(p=0.5, inplace=False),
            Linear(in_features=64, out_features=1, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = self.avgpool(x)
        # print(x.size())

        # flattenning 
        x = x.view(-1,128*4*4)
        # print(x.size())
        logits = self.classifier(x)
        # print(logits.size())
        return logits

def load_trained_model(model='shallow_cnn',model_path='',device=device,verbose=0):
    trained_model = load_model(model,verbose=verbose);
    print("Loading trained weights into current model...")
    trained_model.load_state_dict(torch.load(model_path,map_location=device),strict=True)
    trained_model.to(device)
    trained_model.eval()
    return trained_model

def load_model(model_type='shallow_cnn',verbose=0):
    print("Loading untrained model...")
    if model_type == 'vgg11':
        vgg = adapt_vgg(models.vgg11())
        model = vgg.to(device)
    
    elif model_type == 'vgg11_bn':
        vgg11_bn = adapt_vgg(models.vgg11_bn())
        model = vgg11_bn.to(device)

    elif model_type == 'vgg13_bn':
        vgg13_bn = adapt_vgg(models.vgg13_bn())
        model = vgg13_bn.to(device)
    
    elif model_type == 'vgg13':
        vgg13_bn = adapt_vgg(models.vgg13())
        model = vgg13_bn.to(device)

    elif model_type == 'vgg19_bn':
        vgg19_bn = adapt_vgg(models.vgg19_bn())
        model = vgg19_bn.to(device)

    elif model_type == 'vgg19':
        vgg19 = adapt_vgg(models.vgg19())
        model = vgg19.to(device)

    elif model_type == 'resnet34':
        resnet34 = adapt_resnet(models.resnet34(),linear_features=512)
        model = resnet34.to(device)

    elif model_type == 'resnet50':
        resnet50 = adapt_resnet(models.resnet50(),linear_features=2048)
        model = resnet50.to(device)
    
    elif model_type == 'resnet101':
        resnet101 = adapt_resnet(models.resnet101(),linear_features=2048)
        model = resnet101.to(device)
    
    elif model_type == 'shallow_cnn':
        custom_nn = NeuralNetwork()
        model = custom_nn.to(device)
    elif model_type == 'super_shallow_cnn':
        custom_nn = SuperShallowCNN()
        model = custom_nn.to(device)
    else:
        custom_nn = NeuralNetwork()
        model = custom_nn.to(device)
    if verbose > 0:
      print(model)
      print('')
      count_trainable_parameters(model)
    return model
    
def adapt_resnet(resnet,linear_features = 512):
    resnet.conv1 = Conv2d(1,64, 7, stride=2,padding=3)
    resnet.fc = Sequential(
    Linear(in_features=linear_features, out_features=1000, bias=True),
    ReLU(inplace=True),
    Dropout(p=0.5, inplace=False),
    Linear(in_features=1000, out_features=1, bias=True)
    )
    return resnet

def adapt_vgg(vgg):
    vgg.features[0] = Conv2d(1,64, 3, stride=1,padding=1)
    vgg.classifier[-1] = Linear(in_features=4096, out_features=1,bias=True)
    # vgg.classifier = Sequential(
    # Linear(in_features=7*7*512, out_features=4096, bias=True),
    # ReLU(inplace=True),
    # Linear(in_features=4096, out_features=4096, bias=True),
    # ReLU(inplace=True),
    # Linear(in_features=4096, out_features=1, bias=True)
    # )
    return vgg

def count_trainable_parameters(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print("Total number of trainable parameters:",pp)

def create_adapted_vgg11():
    vgg = models.vgg11()
    vgg.features[0] = Conv2d(1,64, 3, stride=1,padding=1)
    vgg.classifier[0] = Linear(in_features=7*7*512, out_features=2048,bias=True)
    vgg.classifier[3] = Linear(in_features=2048, out_features=2048,bias=True)
    vgg.classifier[-1] = Linear(in_features=2048, out_features=1,bias=True)
    return vgg
