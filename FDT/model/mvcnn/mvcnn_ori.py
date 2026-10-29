import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class VGG11(nn.Module):
    def __init__(self, in_channels=None):
        super(VGG11, self).__init__()
        self.in_channels = in_channels
        self.net1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(in_features=7*10*512,out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096,out_features=4096),
        )


    def forward(self,x):
        print('this is just for backbone')
        return None



class MVCNN_ori(nn.Module):

    def __init__(self, num_classes=1000):
        super(MVCNN_ori, self).__init__()
        self.backbone = VGG11(in_channels=1)
        self.net1 = self.backbone.net1
        self.net2 = self.backbone.net2

        self.dp = nn.Dropout(0.5)
        self.f2c = nn.Linear(4096, num_classes)

    def forward(self, x):
        view_pool = []
        for v in x:
            v = self.net1(v)
            v = v.view(v.size(0), 7*10*512)
            view_pool.append(v)

        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])

        f = self.net2(pooled_view)

        c = self.dp(f)
        c = self.f2c(c)

        return c, f


#######################################for debug this model####################################################
# x = torch.rand((1, 1, 240, 320))
# x = [x, x, x]
# model = MVCNN_ori()
# c, f = model(x)
# print(c.shape)
# print(f.shape)
