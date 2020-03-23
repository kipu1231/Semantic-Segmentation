import torch
import torch.nn as nn
from torchvision import models


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        ''' load pretrained resnetmodel and freeze parameter '''
        model = models.resnet18(pretrained=True)
     
        self.resmodel = torch.nn.Sequential(*(list(model.children())[:-2]))


        ''' declare layers used in this network'''
        # first block
        self.transconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu1 = nn.ReLU() #11x14 --> 22x28

        # second block
        self.transconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU() #22x28 --> 44x56

        # third block
        self.transconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU() #44x56 --> 88x112

        # fourth block
        self.transconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu4 = nn.ReLU() #88x112 --> 176x224

        # fifth block
        self.transconv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu5 = nn.ReLU() #176x224 --> 352x448

        # sixth block
        self.conv6 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True)  # 352x448 --> 352x448


    def forward(self, img):

        x = self.resmodel(img)

        x = self.relu1(self.transconv1(x))

        x = self.relu2(self.transconv2(x))

        x = self.relu3(self.transconv3(x))

        x = self.relu4(self.transconv4(x))

        x = self.relu5(self.transconv5(x))

        x = self.conv6(x)

        return x