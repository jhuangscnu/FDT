import torch.nn as nn
import torch

# basic block of resnet
class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        # for downsampling res
        if self.downsample is not None:
            identity = self.downsample(x)
        # conv 1
        out = self.conv1(x)
        out = self.bn1(out)  
        out = self.relu(out)  
        # conv 2
        out = self.conv2(out)
        out = self.bn2(out)
        # res
        out += identity
        out = self.relu(out)
        return out

# restnet34
class ResNet34backbone(nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet34backbone,self).__init__()
        # for input layer
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,out_channels=64,kernel_size=7,stride=2, padding=3,bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # for layers
        self.layer1 = self.make_layer(in_channel=64, out_channel=64,num_block=3,stride=1)
        self.layer2 = self.make_layer(in_channel=64, out_channel=128,num_block=4,stride=2)
        self.layer3 = self.make_layer(in_channel=128, out_channel=256,num_block=6,stride=2)
        self.layer4 = self.make_layer(in_channel=256, out_channel=512,num_block=3,stride=2)

        # for feature
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # this should be delete as restnet34 is just a backbone of mvcnn
        # self.fc = nn.Linear(512, num_classes)

    def make_layer(self,in_channel, out_channel,num_block,stride=1):
        downsample = None
        if stride==2:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                          kernel_size=1,stride=2,bias=False),
                nn.BatchNorm2d(out_channel)
            )
        layers=[]
        # concate layers
        layers.append(ResidualBlock(in_channel=in_channel,out_channel=out_channel,
                                    downsample=downsample,stride=stride))
        
        for i in range(1,num_block):
            layers.append(ResidualBlock(in_channel=out_channel,out_channel=out_channel,stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x


class MVCNN_imp(nn.Module):

    def __init__(self, num_classes=1000):
        super(MVCNN_imp, self).__init__()
        self.net1 = ResNet34backbone()
        self.net2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        view_pool = []
        for v in x:
            v = self.net1(v)
            v = v.view(v.size(0), -1)
            view_pool.append(v)

        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            f = torch.max(pooled_view, view_pool[i])

        c = self.net2(f)

        return c, f







#######################################for debug this model####################################################
# x = torch.rand((1, 1, 240, 320))
# x = [x, x, x]
# model = MVCNN_imp()
# c, f = model(x)
# print(c.shape)
# print(f.shape)
