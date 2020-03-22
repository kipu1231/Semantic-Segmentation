import torch
import torch.nn as nn
from torchvision import models


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        ''' load pretrained resnetmodel and freeze parameter '''
        model = models.resnet18(pretrained=True)
        # for param in model.parameters():
        #      param.requires_grad = False

        #self.resmodel = nn.Sequential(model)
        self.resmodel = torch.nn.Sequential(*(list(model.children())[:-2]))


        ''' declare layers used in this network'''
        #Hintergrundwissen
        #Convolution:
        # - fasst Pixelbereiche zusammen, da sich immer Pixel einer Region (bestimmt durch Filtergröße) angeschaut werden
        # - gibt es mehrere Filter, so wird jede Bildregion von jedem Filter einmal angeschaut
        # - Padding: fügt am Rand noch eine Zahl hinzu, sodass sich die Größe des Outputs nicht ändern (Annahme: stride = 1)
        # - Stride: gibt an wie groß die Schritte sind, die Filter geht (da hier stride = 2 wird Größe halbiert)
        # - Berechnung der Output Size = (input_size - filter)/stride + 1
        # --> Ist ein lokaler lineare Operator

        #Aktivierungsfunktion (hier Relu)
        # - Ziel ist es Nichtlinearität ins Netz einzuführen (da die meisten Probleme nicht-linear sind)
        # - würde man darauf verzichten, würde das Netz wie ein Netz mit einem Layer arbeiten

        #Pooling
        # - macht die Repräsentation kleiner und damit einfacher zu handeln, wird für jede Feature Map einzeln angewendet

        #Fully Connected Layer (FC)
        # - wird am Ende für die Klassifikation verwendet
        # - verbindet alle Neuronen und mapt diese auf die Anzahl des Outputs

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

        # classification
        #o = (Activation('softmax'))(o)

        #self.pool7 = nn.MaxPool2d()
        #self.pool7 = nn.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
        # ## (None, 7, 7, 512)

        # self.avgpool = nn.AvgPool2d(16)
        # self.fc = nn.Linear(64, 4)
        #self.avgpool = nn.AvgPool2d(8)
        #self.fc = nn.Linear(128, 9)

    def forward(self, img):

        x = self.resmodel(img)

        x = self.relu1(self.transconv1(x))

        x = self.relu2(self.transconv2(x))

        x = self.relu3(self.transconv3(x))

        x = self.relu4(self.transconv4(x))

        x = self.relu5(self.transconv5(x))

        x = self.conv6(x)

        return x