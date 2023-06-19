import os
import random  # to set the python random seed
import numpy as np  # to set the numpy random seed
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 自定义的Dataloader,网络和损失函数
from cwru_processing import CWRU_DataLoader
from loss_func import ParamCrossEntropy, FairLoss
from trainer import WarmUpWithMultistepLr, train, test
from Model.ResNet1d import create_ResNetCWRU
from Model.dot_product_classifier import DotProduct_Classifier

# 记录参数并上传wandb
from utils.metric import *
import wandb
import logging

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

def main(config):
    # 配置训练模型时使用的设备(cpu/cuda)
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 如果使用cuda则修改线程数和允许加载数据到固定内存
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    # 设置随机数种子，保证结果的可复现性
    random.seed(config.seed)  # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    np.random.seed(config.seed)  # numpy random seed
    # 固定返回的卷积算法，保证结果的一致性
    torch.backends.cudnn.deterministic = True

    # 读取数据集
    path = r'Data\48k_DE\HP3'
    low_loader, up_loader, test_loader = CWRU_DataLoader(d_path=path,
                                                            length=400,
                                                            step_size=200,
                                                            train_number=1800,
                                                            test_number=300,
                                                            valid_number=300,
                                                            batch_size=60,
                                                            normal=True,
                                                            IB_rate=config.IB_rate,
                                                            transforms_name=config.transforms,
                                                            use_sampler=config.use_sampler,
                                                            val_unbal=False)

    # 定义使用的网络模型（特征提取器和分类器）
    model_dict = {}
    model_dict['Backbone'] = eval(config.model_names['Backbone'])\
                                (**config.model_params['Backbone']).to(device)
    model_dict['Classifier'] = eval(config.model_names['Classifier'])\
                                (**config.model_params['Classifier']).to(device)
    model = nn.Sequential(model_dict['Backbone'], model_dict['Classifier'])

    # 定义损失函数
    lossfn_dict = {}
    lossfn_dict['train_loss'] = eval(config.lossfn_names['train_loss'])\
                                        (**config.lossfn_params['train_loss']).to(device)
    lossfn_dict['val_loss'] = eval(config.lossfn_names['val_loss'])\
                                        (**config.lossfn_params['val_loss']).to(device)

    # 定义训练模型的优化器
    low_optimizer = optim.SGD(model.parameters(), **config.low_optim_params)
    # 定义训练损失函数参数的优化器
    dy = np.ones([config.num_classes]) * 0.
    dy = torch.tensor(dy, dtype=torch.float32, device=device, requires_grad=True)
    ly = np.ones([config.num_classes]) * 0.
    ly = torch.tensor(ly, dtype=torch.float32, device=device, requires_grad=True)
    num_augmentations = len(config.augmentations)  # 数据增强的数量
    py = np.ones([num_augmentations, config.num_classes]) * 0.
    py = torch.tensor(py, dtype=torch.float32, device=device, requires_grad=True)
    uy = np.ones([num_augmentations, config.num_classes]) * 0.
    uy = torch.tensor(uy, dtype=torch.float32, device=device, requires_grad=True)
    up_optimizer = optim.SGD(params=[{'params': dy}, {'params': ly}, {'params': py}, {'params': uy}],
                            **config.up_optim_params)
    # 优化器的学习率策略
    warm_up_with_multistep_lr_low = WarmUpWithMultistepLr(**config.low_lr_schedule)
    warm_up_with_multistep_lr_up = WarmUpWithMultistepLr(**config.up_lr_schedule)
    low_lr_scheduler = optim.lr_scheduler.LambdaLR(
        low_optimizer, lr_lambda=warm_up_with_multistep_lr_low)
    up_lr_scheduler = optim.lr_scheduler.LambdaLR(
        up_optimizer, lr_lambda=warm_up_with_multistep_lr_up)
    optimizer_dict = {'low_optimizer': low_optimizer, 'up_optimizer': up_optimizer}

    # 追踪模型参数并上传wandb
    wandb.watch(model_dict['Backbone'], log="all")
    wandb.watch(model_dict['Classifier'], log="all")

    # 设置期望正确率
    acc = 0

    # 定义存储模型参数的文件名
    model_paths = {'Backbone': config.datasets + '_' + config.model_names['Backbone'] + '.pth',
                'Classifier': config.datasets + '_' + config.model_names['Classifier'] + '.pth'}

    # 训练模型
    for epoch in range(1, config.epochs + 1):
        acc = train(model, device, low_loader=low_loader,
                    up_loader=up_loader, optimizer_dict=optimizer_dict,
                    lossfn_dict=lossfn_dict, epoch=epoch, epochs=config.epochs,
                    params=[dy, ly, py, uy], transforms_name=config.augmentations,
                    transforms_temperature=config.temperature, num_classes=config.num_classes)

        low_lr_scheduler.step()
        up_lr_scheduler.step()

        # 如果acc为非数，则终止训练
        if np.any(np.isnan(acc)):
            print("NaN")
            break

        # 每训练完一轮,测试模型的指标
        test(model, device, test_loader)

    # 如果正确率非空,则保存模型到wandb
    if not np.any(np.isnan(acc)):
        wandb.save('*.pth')
    # 如果正确率非空,训练结束存储模型参数
    if not np.any(np.isnan(acc)):
        torch.save(model_dict['Backbone'].state_dict(),
                os.path.join(wandb.run.dir, model_paths['Backbone']))
        torch.save(model_dict['Classifier'].state_dict(),
                os.path.join(wandb.run.dir, model_paths['Classifier']))

    dy_log = open(f'dy_IB{config.IB_rate[0]}.txt', mode='w')
    ly_log = open(f'ly_IB{config.IB_rate[0]}.txt', mode='w')
    py_log = open(f'py_IB{config.IB_rate[0]}.txt', mode='w')
    uy_log = open(f'uy_IB{config.IB_rate[0]}.txt', mode='w')
    dy_log.write(f'{list(dy.detach().cpu().numpy())}\n')
    ly_log.write(f'{list(ly.detach().cpu().numpy())}\n')
    py = list(py.detach().cpu().numpy())
    py = [list(i) for i in py]
    py_log.write(f'{list(py)}\n')
    uy = list(uy.detach().cpu().numpy())
    uy = [list(i) for i in uy]
    uy_log.write(f'{list(uy)}\n')
    dy_log.close()
    ly_log.close()
    py_log.close()
    uy_log.close()

if __name__ == '__main__':
    # 定义wandb上传项目名
    IB_rate = [100, 1]
    name = 'IB=' + str(IB_rate[0]) + ':' + str(IB_rate[1])
    wandb.init(project="autobalance_loss_and_aug(loss_search)", name=name)
    wandb.watch_called = False

    # 定义上传的超参数
    config = wandb.config
    # 数据集及其预处理
    config.datasets = 'CWRU(48k_DE_HP3)'  # 数据集
    config.transforms = ()  # 数据增强
    config.batch_size = 60  # 批量样本数
    config.IB_rate = IB_rate
    config.use_sampler = False  # 使用采样器
    # config.sampler = "ClassAwareSampler"  # 训练集采样器
    # config.num_sampler_cls = 6  # 采样器重复采样某一类别样本次数
    config.test_batch_size = 60  # 测试批量样本数
    config.num_classes = 10  # 样本类别数
    config.augmentations = ('AddGaussian', 'MaskNoise', 'RandomScale', 'Translation')
    config.temperature = 0.05

    # 网络模型的参数
    config.model_names = {'Backbone': 'create_ResNetCWRU', 'Classifier': 'DotProduct_Classifier'}  # 网络模型
    ResNetCWRU_params = {'seq_len': 400, 'num_blocks': 2, 'planes': [10, 10, 10],
                                'kernel_size': 10, 'pool_size': 2, 'linear_plane': 100}
    DotProduct_Classifier_params = {'linear_plane': 100, 'num_classes': 10}
    config.model_params = {'Backbone': ResNetCWRU_params,
                        'Classifier': DotProduct_Classifier_params}  # 网络模型的相关参数

    # 损失函数的参数
    config.lossfn_names = {'train_loss': 'ParamCrossEntropy',
                        'val_loss': 'FairLoss'}  # 损失函数
    # config.lossfn_names = {'train_loss': 'ParamCrossEntropy',
    #                        'val_loss': 'nn.CrossEntropyLoss'}  # 损失函数
    ParamCrossEntropy_params = {}
    FairLoss_params = {'lamda': 0.1}
    config.lossfn_params = {'train_loss': ParamCrossEntropy_params,
                            'val_loss': FairLoss_params}  # 损失函数的参数

    # 训练的相关参数
    config.epochs = 300  # 训练轮数
    config.low_lr = 0.1  # 底层学习率(训练模型的学习率)
    config.up_lr = 0.01  # 顶层学习率(调整损失函数超参数的学习率)
    config.momentum = 0.9  # 动量系数
    config.model_weight_decay = 0.0005  # 权重衰减
    config.params_weight_decay = 0.
    config.optimizer = 'SGD'
    # 训练网络模型的优化器参数
    config.low_optim_params = {'lr': config.low_lr, 'momentum': config.momentum,
                                'weight_decay': config.model_weight_decay}
    # 训练损失函数超参数的优化器参数
    config.up_optim_params = {'lr': config.up_lr, 'momentum': config.momentum,
                            'weight_decay': config.params_weight_decay}
    # 学习率策略参数
    config.start_epoch = 0
    config.low_lr_schedule = {'start_epoch': 0, 'lr_warmup': 5,
                            'lr_schedule': [220, 260], 'lr_multiplier': [1, 0.1, 0.01]}
    config.up_lr_schedule = {'start_epoch': config.start_epoch, 'lr_warmup': 10,
                            'lr_schedule': [220, 260], 'lr_multiplier': [1, 0.1, 0.01]}

    config.no_cuda = False  # 不使用cuda(T/F)

    config.seed = 42  # 随机数种子

    main(config)

