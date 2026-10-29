import torch
import torch.nn as nn
import numpy as np
# from torchsummary import summary
import time
import sys
import os

# sys.path.append('..')
# from params import args
# from utils import batch_cos_distance


class CenterLoss(nn.Module):
    """
    Center loss:

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    args:
        num_classes(int): numbers of classes.
        feat_dim(int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim=256):
        super(CenterLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # center point
        # self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        if torch.cuda.is_available():
            self.centers.cuda()

        # centers learning rate
        self.lr = nn.Parameter(torch.Tensor([0.001]))

    # forward compute
    # get loss and update centers
    def forward(self, x, labels):
        """
        args:
            x: feature matrix with shape(batch size, feat dim).
            labels: ground truth labels with shape(batch size).
        """
        batchSize = x.size(0)
        feat_dim = x.size(1)

        centerBatch = self.centers.index_select(0, labels.long())#.cuda()
        if torch.cuda.is_available():
            centerBatch.cuda()

        # 计算中心和feature的距离
        diff = centerBatch - x
        loss = (x - centerBatch).pow(2).sum() / 2.0 / batchSize

        # 计算更新值
        counts = self.centers.new_ones(self.centers.size(0))#.cuda()
        ones = self.centers.new_ones(labels.size(0))#.cuda()
        gradCenters = self.centers.new_zeros(self.centers.size())#.cuda()
        if torch.cuda.is_available():
            counts.cuda()
            ones.cuda()
            gradCenters.cuda()

        # 按label中的索引计算类别出现次数
        counts = counts.scatter_add_(0, labels.long(), ones)

        # 将矩阵reshape为与特征向量一致
        gradCenters.scatter_add_(0, labels.unsqueeze(1).expand(x.size()).long(), diff)

        # 计算更新值
        self.centers = nn.Parameter(self.centers - self.lr * gradCenters)#.cuda()
        if torch.cuda.is_available():
            self.centers.cuda()

        return loss

    # (unused) initialize by user
    def _center_initialize(self, genNum=200):
        centers = torch.zeros(self.num_classes, self.feat_dim)#.cuda()
        if torch.cuda.is_available():
            centers.cuda()
        for i in range(self.num_classes):
            maxEuclideanDist = 0
            for j in range(genNum):
                randFeat = torch.randn(self.feat_dim)#.cuda()
                if torch.cuda.is_available():
                    randFeat.cuda()

                euclideanDist = (randFeat - centers[: j + 1]) ** 2
                euclideanDist = euclideanDist.sum(dim=1, keepdim=True)
                euclideanDist = torch.sqrt(euclideanDist).sum(dim=0, keepdim=True)

                if euclideanDist > maxEuclideanDist:
                    maxEuclideanDist = euclideanDist
                    centers[i] = randFeat
        return nn.Parameter(centers)


# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     loss = CenterLoss(num_classes=664).cuda()
#
#     testData = torch.randn(8, 256).to(args.device)
#     testLabel = torch.Tensor(np.random.rand(0, 664, size=(8))).long().cuda()
#
#     out = loss(testData, testLabel)
#     print(out)
