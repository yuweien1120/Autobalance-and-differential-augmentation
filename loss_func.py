"""设计超参数可调整的损失函数"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from cwru_processing import CWRU_DataLoader

class ParamCrossEntropy(nn.Module):
    def __init__(self):
        super(ParamCrossEntropy, self).__init__()

    def forward(self, logits, targets, params):
        """
        计算参数化的交叉熵损失函数
        :param logits: 分类器输出的值(shape[batch_size, num_classes]))
        :param targets: 样本的标签(shape[batch_size])
        :param params: {dy, ly}损失函数的参数(shape[2, num_classes])
        :return: 损失函数值(scalar)
        """
        dy = params[0]
        ly = params[1]
        x = logits * torch.sigmoid(6. * dy) + ly
        loss = F.cross_entropy(x, targets)
        return loss

class CrossEntropyBal(nn.Module):
    def __init__(self):
        super(CrossEntropyBal, self).__init__()

    def forward(self, logits, targets, num_classes, device):
        """
        计算平衡的交叉熵损失函数(相当于先求每一类的损失函数的均值，再求所有类的损失函数的均值的均值)
        :param num_classes: 类别数(scalar)
        :param logits: 分类器输出的值(shape[batch_size, num_classes])
        :param targets: 样本的标签(shape[batch_size])
        :return: 损失函数值(scalar)
        """
        num_samples = len(logits)
        count = torch.bincount(targets, minlength=num_classes)
        weight = torch.tensor([1 / i if i != 0 else 0 for i in count])
        weight = num_samples / num_classes * weight
        weight = weight.to(device)
        CE = nn.CrossEntropyLoss(weight=weight)
        loss = CE(logits, targets)
        return loss

class FairLoss(nn.Module):
    def __init__(self, lamda):
        super(FairLoss, self).__init__()
        self.lamda = lamda

    def forward(self, logits, targets, num_classes, device):
        """
        计算公平交叉熵损失函数(L_fair = (1 - lamda) * L_CE + lamda * L_CE_bal)
        :param num_classes:
        :param logits: 分类器输出的值(shape[batch_size, num_classes])
        :param targets: 样本的标签(shape[batch_size])
        :return: 损失函数值(标量)
        """
        lossfun1 = nn.CrossEntropyLoss()
        lossfun2 = CrossEntropyBal()
        loss1 = lossfun1(logits, targets)
        loss2 = lossfun2(logits, targets, num_classes, device)
        loss_fair = (1 - self.lamda) * loss1 + self.lamda * loss2
        return loss_fair



if __name__ == "__main__":
    logits = torch.rand([10, 10], requires_grad=True)
    weight = torch.rand([10], requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1, 4, 5, 6, 8, 9, 3])
    loss = F.cross_entropy(logits, targets, weight=weight)





