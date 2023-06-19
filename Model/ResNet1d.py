import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, kernel_size):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(planes)
        )

    def forward(self, x):
        length = x.size()[-1]

        out = F.relu(self.bn1(self.conv1(x)))
        out = out[:, :, :length]
        out = self.bn2(self.conv2(out))
        out = out[:, :, :length]

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetCWRU(nn.Module):
    def __init__(self, block, seq_len, num_blocks, planes, kernel_size, pool_size, linear_plane):
        super(ResNetCWRU, self).__init__()
        self.seq_len = seq_len
        self.planes = planes
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(1, self.planes[0], kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(self.planes[0])
        self.layer = self._make_layer(block, num_blocks, pool_size)

        self.linear = nn.Linear(self.planes[-1] * (self.seq_len // (pool_size ** (num_blocks - 1))), linear_plane)

    def _make_layer(self, block, num_blocks, pool_size):
        layers = []

        for i in range(num_blocks - 1):
            layers.append(block(self.planes[i], self.planes[i + 1], kernel_size=self.kernel_size))
            layers.append(nn.MaxPool1d(pool_size))

        layers.append(block(self.planes[-2], self.planes[-1], kernel_size=self.kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = out[:, :, :self.seq_len]
        out = self.layer(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear(out))

        return out


def create_ResNetCWRU(seq_len, num_blocks, planes, kernel_size, pool_size, linear_plane):
    net = ResNetCWRU(BasicBlock, seq_len, num_blocks, planes, kernel_size, pool_size, linear_plane=linear_plane)
    return net

if __name__ == '__main__':
    from torchsummary import summary
    import copy
    import torch.optim as optim
    from dot_product_classifier import DotProduct_Classifier
    from cwru_processing import CWRU_DataLoader
    from torch.autograd import grad
    from differential_augmentation import AddGaussian

    ResNetCWRU_params = {'seq_len': 400, 'num_blocks': 2, 'planes': [10, 10, 10],
                        'kernel_size': 10, 'pool_size': 2, 'linear_plane': 100}
    feat = create_ResNetCWRU(**ResNetCWRU_params)
    classifier = DotProduct_Classifier(100, 10)
    net = nn.Sequential(feat, classifier)

    path = r'C:\Users\0\Desktop\故障诊断\Autobalance_finite_difference\Data\48k_DE\HP3'
    transforms_name = ()

    train_loader, valid_loader, test_loader = CWRU_DataLoader(d_path=path,
                                                              length=400,
                                                              step_size=200,
                                                              train_number=1800,
                                                              test_number=300,
                                                              valid_number=300,
                                                              batch_size=60,
                                                              normal=True,
                                                              IB_rate=[10, 1],
                                                              transforms_name=transforms_name,
                                                              use_sampler=False,
                                                              val_unbal=True)
    for i, j in train_loader:
        input_data = i
        target = j
        break
    p = torch.tensor(0.5, requires_grad=True)
    aug = AddGaussian(p)
    input_data1 = aug(input_data)
    input_data1 = input_data1.reshape(input_data1.shape[0], 1, input_data1.shape[1])

    loss_fn = nn.CrossEntropyLoss()


    # for param in net.parameters():
    #     print('net_parameter2: %s' % param)
    # for param in model.parameters():
    #     print('model_parameter: %s' % param)


    optimizer = optim.SGD([{'params': feat.parameters(), 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005},
                        {'params': classifier.parameters(), 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005}])


    preds = net(input_data1)
    loss1 = loss_fn(preds, target)
    loss1.backward(retain_graph=True)
    print(p.grad)
    # optimizer.step()
    # p.grad = torch.tensor(0.)
    # input_data1 = aug(input_data)
    # input_data1 = input_data1.reshape(input_data1.shape[0], 1, input_data1.shape[1])
    preds = net(input_data1)
    loss2 = loss_fn(preds, target)
    loss2.backward()
    print(p.grad)

    # input_data1 = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
    # input_data1 += p
    #
    # preds = net(input_data1)
    # loss1 = loss_fn(preds, target)
    # loss1.backward()
    # print(p.grad)


    # print(input_data)
    # v = [g.grad.data.detach() for g in net.parameters()]
    # print(v[0][0])
    #
    # for param in net.parameters():
    #     param.data.add_(1.)
    #
    # preds = net(input_data)
    # loss2 = loss_fn(preds, target)
    # loss2.backward()
    # v = [g.grad.data.detach() for g in net.parameters()]
    # print(v[0][0])




    # print('loss: %.2f' % loss1)
    # optimizer.step()


    # v = [g.grad.data.detach() for g in net.parameters()]
    # print(v[0][0])
    # print('loss: %.2f' % loss)
    # optimizer.step()

    # model = copy.deepcopy(net)
    # optim_temp_params_dict = copy.deepcopy(optimizer.state_dict())
    # for param in optim_temp_params_dict['param_groups']:
    #     del param['params']
    # optimizer_temp = optim.SGD(model.parameters(), lr=0.1)
    # print(optimizer.state_dict())
    # optimizer_temp.load_state_dict(optimizer.state_dict())
    #
    # print(optimizer.state_dict())
    # print(optimizer_temp.state_dict())

    # optimizer.zero_grad()
    # preds = net(input_data)
    # loss = loss_fn(preds, target)
    # loss.backward()
    # optimizer.step()
    # print("loss1: %f" % loss)

    # optimizer_temp.zero_grad()
    # preds = model(input_data)
    # loss = loss_fn(preds, target)
    # loss.backward()
    # optimizer_temp.step()
    # print("loss2: %f" % loss)

    # optimizer.zero_grad()
    # preds = net(input_data)
    # loss = loss_fn(preds, target)
    # print("loss3: %f" % loss)

    # optimizer_temp.zero_grad()
    # preds = model(input_data)
    # loss = loss_fn(preds, target)
    # print("loss4: %f" % loss)
    # loss.backward()
    # optimizer_temp.step()
    #
    # preds = model(input_data)
    # loss = loss_fn(preds, target)
    # print("loss5: %f" % loss)
    #
    # optimizer.zero_grad()
    # preds = net(input_data)
    # loss = loss_fn(preds, target)
    # print("loss6: %f" % loss)


