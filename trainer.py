import os
import random  # to set the python random seed
import numpy as np # to set the numpy random seed
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.autograd import grad
from pycm import *
from matplotlib import pyplot as plt
from differential_augmentation import *
import copy

# 记录参数并上传wandb
from utils.metric import AverageMeter, accuracy
import wandb
import logging

class WarmUpWithMultistepLr:
    # 在warmup阶段，参数更新的学习率变化策略
    def __init__(self, start_epoch, lr_warmup, lr_schedule: list, lr_multiplier: list):
        """
        初始化学习率变化策略的参数
        :param start_epoch: 学习率开始的epoch
        :param lr_warmup: 开始阶段学习率线性上升的epoch,上升的斜率为1/low_lr_warmup
        :param lr_schedule: 学习率减小的转折点(list)
        :param lr_multiplier: 不同阶段的学习率系数(list)
        """
        self.lr_warmup = lr_warmup
        self.lr_schedule = lr_schedule
        self.lr_multiplier = lr_multiplier
        self.start_epoch = start_epoch

    def __call__(self, epoch):
        if self.start_epoch <= epoch < (self.start_epoch + self.lr_warmup):
            return (epoch - self.start_epoch + 1) / self.lr_warmup
        else:
            return self.lr_multiplier[len([m for m in self.lr_schedule if m <= epoch])]

def gather_flat_grad(loss_grad):
    # 将所有的梯度信息展开成1维
    return torch.cat([p.contiguous().view(-1) for p in loss_grad if not p is None])

def assign_hyper_gradient(params, gradient):
    # 把超参数的梯度信息存进去
    i = 0
    for para in params:
        if para.requires_grad:
            num = para.nelement()  # 统计元素个数
            grad = gradient[i:i+num].clone()
            grad = torch.reshape(grad, para.shape)
            para.grad = grad
            i += num

def train(model, device, low_loader, up_loader, optimizer_dict, lossfn_dict,
        epoch, epochs, params, transforms_name, transforms_temperature, num_classes, is_search=True):
    """
    对模型进行一轮训练，并打印和上传相关的训练指标
    训练指标包括：训练标签位于模型输出前1的正确率，训练标签位于模型输出前5的正确率，训练的损失值
    :param num_classes: 样本类别数
    :param model: 网络模型
    :param device: 训练使用的设备,cuda或cpu
    :param low_loader: 训练集
    :param up_loader: 验证集
    :param optimizer_dict: 多个优化器(字典索引)
    :param lossfn_dict: 训练损失函数和验证损失函数(字典索引)
    :param epoch: 训练轮数
    :param epochs: 训练总轮数
    :param params: 训练损失函数的超参数
    :param transforms_name: 数据增强的函数名(tuple)
    :param transforms_temperature: 松弛化的温度系数
    :param is_search: 超参数搜索阶段
    :return: 训练标签位于模型输出前1的正确率
    """
    model.train()  # 模型为训练模式
    train_loss = AverageMeter()  # 统计训练损失
    train_top1 = AverageMeter()  # 统计训练top1准确率
    train_top5 = AverageMeter()  # 统计训练top5准确率
    low_optimizer = optimizer_dict['low_optimizer']  # 更新模型参数的优化器
    up_optimizer = optimizer_dict['up_optimizer']  # 更新超参数的优化器
    low_lossfn = lossfn_dict['train_loss']  # 训练损失函数
    up_lossfn = lossfn_dict['val_loss']  # 验证损失函数
    num_theta = sum(p.numel() for p in model.parameters())  # 模型参数(theta)的数量

    if is_search:
        print('up_level. ', 'low_lr: %.6f, up_lr: %.6f' % (low_optimizer.param_groups[0]['lr']
                                                        , up_optimizer.param_groups[0]['lr']))
        val_loss = AverageMeter()

    for batch_idx, (low_data, low_target) in enumerate(low_loader):
        # 先计算验证损失对超参数的导数
        low_data, low_target = low_data.to(device), low_target.to(device)  # 将训练集数据迁移到gpu上
        # 如果是超参数搜索模式,求验证损失对超参数的导数
        if is_search:
            # 对训练集数据做数据增强
            low_data_aug = data_augment(low_data, low_target, params, transforms_name, transforms_temperature)
            # 将增强后的数据添加通道维度
            low_data_aug = low_data_aug.reshape(low_data_aug.shape[0], 1, low_data_aug.shape[1])

            # 计算d_val_loss_d_theta'(theta':利用训练损失更新后的模型参数)
            up_optimizer.zero_grad()  # 超参数优化的梯度清零
            model_temp = copy.deepcopy(model)  # 创建临时模型(用于计算验证损失对模型参数的梯度)
            model_temp.train()  # 设置临时模型为训练模式
            low_optimizer_temp = optim.SGD(params=model_temp.parameters(), lr=0.1)  # 定义临时模型的优化器(用于后续更新临时模型参数)
            low_optimizer_temp.load_state_dict(low_optimizer.state_dict())  # 加载模型参数以及momentu的缓存
            # 计算训练损失然后更新临时模型参数(得到theta')
            low_preds = model_temp(low_data_aug)
            low_loss = low_lossfn(low_preds, low_target, params)
            low_optimizer_temp.zero_grad()
            low_loss.backward(retain_graph=True)  # 保持计算图,因为后面还要继续用low_data_aug,不保持的话数据增强部分没法backward
            low_optimizer_temp.step()
            # 计算d_val_loss_d_theta'(平均值)
            up_batch_num = len(up_loader)
            for up_data, up_target in up_loader:
                up_data, up_target = up_data.to(device), up_target.to(device)  # 将验证集数据迁移到gpu上
                up_data = up_data.reshape(up_data.shape[0], 1, up_data.shape[1])
                up_preds = model_temp(up_data)
                up_loss = up_lossfn(up_preds, up_target, num_classes, device)  # 计算验证损失,由于计算的是平均值,所以先除计算的batch
                # 更新统计验证损失
                with torch.no_grad():
                    val_loss.update(up_loss.item(), n=up_target.size(0))
                up_loss.backward()  # 累积模型参数的梯度
            # 获取d_val_loss_d_theta,list中的元素对应model.parameters()迭代返回的参数的梯度
            d_val_loss_d_theta = [param.grad.data.detach() / float(up_batch_num) for param in model_temp.parameters()]
            # 删除临时模型的优化器
            del low_optimizer_temp
            model_temp = copy.deepcopy(model)  # 再次创建临时模型(这次用于计算w+和w-时的训练损失)
            model_temp.train()
            r = 1e-2
            R = r / gather_flat_grad(d_val_loss_d_theta).data.detach().norm()  # 近似用的小标量
            # 将临时模型的参数变为w+
            for p, v in zip(model_temp.parameters(), d_val_loss_d_theta):
                p.data.add_(R, v)
            low_preds = model_temp(low_data_aug)
            low_loss = low_lossfn(low_preds, low_target, params)
            up_optimizer.zero_grad()  # 超参数梯度清零
            # 计算w+时超参数的训练损失梯度,并转为向量,这里仍然要保持计算图,因为算w-的损失时候还要用low_data_aug
            d_train_loss_plus_d_alpha = gather_flat_grad(grad(low_loss, params, retain_graph=True))
            # 将临时模型的参数变为w-
            for p, v in zip(model_temp.parameters(), d_val_loss_d_theta):
                p.data.sub_(2. * R, v)
            low_preds = model_temp(low_data_aug)
            low_loss = low_lossfn(low_preds, low_target, params)
            up_optimizer.zero_grad()
            # 计算w-时超参数的训练损失梯度,并转为向量
            d_train_loss_sub_d_alpha = gather_flat_grad(grad(low_loss, params))
            low_lr = low_optimizer.state_dict()['param_groups'][0]['lr']  # 获取模型参数的学习率
            # 获取超参数的验证损失梯度
            dval_loss_d_alpha = - low_lr * (d_train_loss_plus_d_alpha - d_train_loss_sub_d_alpha) / (2 * R)
            up_optimizer.zero_grad()
            assign_hyper_gradient(params, dval_loss_d_alpha)
            up_optimizer.step()  # 更新超参数

        # 更新模型参数
        # 用(更新后的)超参数对训练集数据做数据增强
        low_data_aug_new = data_augment(low_data, low_target, params, transforms_name, transforms_temperature)
        # 将数据添加通道维度
        low_data_aug_new = low_data_aug_new.reshape(low_data_aug_new.shape[0], 1, low_data_aug_new.shape[1])
        low_preds = model(low_data_aug_new)
        low_loss = low_lossfn(low_preds, low_target, params)
        low_optimizer.zero_grad()
        low_loss.backward()
        low_optimizer.step()

        # 更新损失,准确率
        with torch.no_grad():
            prec1, prec5 = accuracy(low_preds, low_target, topk=(1, 5))
            train_loss.update(low_loss.item(), n=low_target.size(0))
            train_top1.update(prec1.item(), n=low_target.size(0))
            train_top5.update(prec5.item(), n=low_target.size(0))

        # 判断loss当中是不是有元素非空，如果有就终止训练，并打印梯度爆炸
        if np.any(np.isnan(low_loss.item())):
            print("Gradient Explore")
            break
        # 每训练20个小批量样本就打印一次训练信息
        if batch_idx % 20 == 0:
            print('Epoch: [%d|%d] Step:[%d|%d], LOW_LOSS: %.5f' %
                (epoch, epochs, batch_idx + 1, len(low_loader), low_loss.item()))
    if is_search:
        print('Epoch: [%d|%d] , UP_LOSS: %.5f' %
            (epoch, epochs, val_loss.avg))
        wandb.log({
            "Train top 1 Acc": train_top1.avg,
            "Train top5 Acc": train_top5.avg,
            "Train Loss": train_loss.avg,
            "Val Loss": val_loss.avg}, commit=False)
        for i, (dy, ly) in enumerate(zip(params[0], params[1])):
            d_index = 'dy' + str(i)
            l_index = 'ly' + str(i)
            wandb.log({d_index: dy, l_index: ly}, commit=False)
        for i, (py, uy) in enumerate(zip(params[2], params[3])):
            for j, (pi, ui) in enumerate(zip(py, uy)):
                p_index = 'py' + str(i) + '_' + str(j)
                u_index = 'uy' + str(i) + '_' + str(j)
                wandb.log({p_index: pi, u_index: ui}, commit=False)
    else:
        wandb.log({
            "Train top 1 Acc": train_top1.avg,
            "Train top5 Acc": train_top5.avg,
            "Train Loss": train_loss.avg}, commit=False)
    return train_top1.avg

def test(model, device, test_loader):
    """
    上传模型在测试集上的测试指标到wandb网页，
    测试指标包括：测试标签位于模型输出前1的正确率，测试标签位于模型输出前5的正确率，测试的损失值
    :param model: 网络模型
    :param device: 训练使用的设备，cuda或cpu
    :param test_loader: 测试训练集
    :return:
    """
    model.eval()
    test_loss = AverageMeter()
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    test_target = torch.tensor([], device=device)
    test_output = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], 1, data.shape[1])
            output = model(data)
            loss = F.cross_entropy(output, target)

            test_target = torch.cat([test_target, target], dim=0)
            test_output.append(output)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            test_top1.update(prec1.item(), n=target.size(0))
            test_top5.update(prec5.item(), n=target.size(0))
            test_loss.update(loss.item(), n=target.size(0))

    _, test_pred = torch.max(torch.vstack(test_output), dim=1)
    cm = ConfusionMatrix(actual_vector=np.array(test_target.long().cpu()),
                        predict_vector=np.array(test_pred.long().cpu()))
    plt.close()
    cm.plot(cmap=plt.cm.Greens,number_label=True)
    wandb.log({
        "Test top 1 Acc": test_top1.avg,
        "Test top5 Acc": test_top5.avg,
        "Test Loss": test_loss.avg,
        'confusion matrix': wandb.Image(plt)})
    # return test_top1.avg 如果是验证集，则可以返回acc

if __name__ == "__main__":
    schedule = WarmUpWithMultistepLr(0, 5, [220, 226], [1, 0.01, 0.001])
    for i in range(300):
        print(schedule(i))