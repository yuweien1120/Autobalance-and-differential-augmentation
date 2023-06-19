from scipy.io import loadmat
import os
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from class_aware_sampler import ClassAwareSampler
from differential_augmentation import AddGaussian, MaskNoise, RandomScale, Translation, \
    data_transform, generate_params_dict, params_index

def CWRU_DataLoader(d_path, length=400, step_size=200, train_number=1800,
                    test_number=300,
                    valid_number=300,
                    batch_size=64,
                    normal=True, IB_rate=(10, 1),
                    transforms_name=("AddGaussian", "RandomScale", "MaskNoise", "Translation"),
                    use_sampler=True,
                    val_unbal=False):
    """对数据进行预处理,返回train_X, train_Y, test_X, test_Y样本.
    :param d_path: 源数据地址
    :param length: 信号长度，默认400
    :param step_size: 信号采样滑窗移动长度，默认200
    :param train_number: 正常样本数(默认1800)
    :param test_number: 测试集的样本数
    :param valid_number: 验证集的样本数
    :param normal: 是否标准化.True,False.默认True
    :param batch_size: 批量大小
    :param IB_rate: 训练集中正常样本和每一个故障样本数的比例
    :param transforms_name: 生成数据增强的方法默认("Add_Gaussian", "Random_Scale", "Mask_Noise", "Translation")
    :param use_sampler: 使用采样器(类感知采样器)
    :param val_unbal: 验证集不平衡(默认False)
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    """

    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)
    # 将文件名列表中结尾不是.mat的文件名去掉，防止下面loadmat报错
    for i in filenames:
        if not i.endswith('.mat'):
            filenames.remove(i)

    def capture():
        """读取mat文件，返回字典
        :return: 数据字典
        """
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    def slice(data, train_number=train_number, test_number=test_number, valid_number=valid_number, val_unbal=val_unbal):
        """
        将数据按数据集的样本数提取
        :param data: 单挑数据
        :param train_number: 训练样本数
        :param test_number: 测试集的样本数
        :param valid_number: 验证集的样本数
        :param val_unbal: 验证集不平衡采样(默认False)
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Valid_Samples = {}
        Test_Samples = {}

        for i in keys:
            slice_data = data[i]
            start = 0
            if "Normal" in i:  # 如果是正常样本，则正常采样
                samp_train = int(train_number)  # 1800
                samp_valid = int(valid_number)  # 300
            else:  # 如果是故障样本，则按IB_rate的比例采样
                samp_train = int(train_number * (IB_rate[1] / IB_rate[0]))  # 180
                if val_unbal:
                    samp_valid = int(valid_number * (IB_rate[1] / IB_rate[0]))
                else:
                    samp_valid = int(valid_number)
            samp_test = int(test_number)
            Train_Sample = []
            Valid_Sample = []
            Test_Sample = []

            # 抓取训练数据
            for j in range(samp_train):
                sample = slice_data[start:start + length]
                start = start + step_size
                Train_Sample.append(sample)
            start = start + step_size

            # 抓取验证数据
            for h in range(samp_valid):
                sample = slice_data[start:start + length]
                start = start + step_size
                Valid_Sample.append(sample)
            start = start + step_size

            # 抓取测试数据
            for h in range(samp_test):
                sample = slice_data[start:start + length]
                start = start + step_size
                Test_Sample.append(sample)
            Train_Samples[i] = Train_Sample
            Valid_Samples[i] = Valid_Sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Valid_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    def scalar_stand(Train_X, Valid_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Valid_X = scalar.transform(Valid_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Valid_X, Test_X

    # 从所有.mat文件中读取出数据的字典
    data = capture()
    # 将数据切分为训练集、测试集
    train, valid, test = slice(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为验证集制作标签，返回X，Y
    Valid_X, Valid_Y = add_labels(valid)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Valid_X, Test_X = scalar_stand(Train_X, Valid_X, Test_X)

    # 需要做一个数据转换，转换成tensor格式.
    # tensor是函数，可以生成指定类型的张量
    # Tensor是类，是默认张量类型torch.FloatTensor()的别名，生成float类型的张量
    Train_X = torch.tensor(Train_X, dtype=torch.float)
    Valid_X = torch.tensor(Valid_X, dtype=torch.float)
    Test_X = torch.tensor(Test_X, dtype=torch.float)
    Train_Y = torch.tensor(Train_Y, dtype=torch.long)
    Valid_Y = torch.tensor(Valid_Y, dtype=torch.long)
    Test_Y = torch.tensor(Test_Y, dtype=torch.long)

    if len(transforms_name) == 0:
        train_dataset = TensorDataset(Train_X, Train_Y)
    else:
        # 生成数据增强的数据
        transforms_params_dict = generate_params_dict(transforms_name)
        Transformed_data1 = data_transform(Train_X, transforms_params_dict)
        Transformed_data2 = data_transform(Train_X, transforms_params_dict)
        # 训练集包含了原样本和标签，还有两个经过数据增强后的样本，用于后面训练时计算对比损失
        train_dataset = TensorDataset(Train_X, Train_Y, Transformed_data1, Transformed_data2)


    valid_dataset = TensorDataset(Valid_X, Valid_Y)
    test_dataset = TensorDataset(Test_X, Test_Y)
    # 是否使用采样器平衡长尾样本
    if use_sampler:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                sampler=ClassAwareSampler(Train_Y, 6))
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    path = r'Data\48k_DE\HP3'
    transforms_name = ("AddGaussian", "RandomScale", "MaskNoise", "Translation")  # 不使用数据增强
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
    for batch, (i, j, k, l) in enumerate(train_loader):
        print(batch, i, j, k, l)
