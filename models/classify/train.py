# -*- coding: utf-8 -* -
from __future__ import print_function
from __future__ import division
import time
import copy
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot
import matplotlib.pyplot as plt

import inference
import pretrainedmodels
from dataset import MyDataset
import pretrainedmodels.utils as utils

pyplot.switch_backend('agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate(net, data_loader, set_name, classes_name):
    """
    对一批数据进行预测，返回混淆矩阵以及Accuracy
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'tesst
    :param classes_name:
    :return:
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for data in data_loader:
        images, labels = data
        images = Variable(images)
        labels = Variable(labels)

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        outputs.detach_()

        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i].cpu().numpy()
            pre_i = predicted[i].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i],
            np.sum(conf_mat[i, :]),
            conf_mat[i, i],
            conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
            conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


def show_confmat(confusion_mat, classes, set_name, out_dir):
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png'))
    plt.close()


def load_data(model):
    trainTransform = utils.TransformImage(model)
    validTransform = utils.TransformImage(model)

    # 构建MyDataset实例
    train_data = MyDataset(data_path=os.path.join(data_dir, 'train'), transform=trainTransform)
    valid_data = MyDataset(data_path=os.path.join(data_dir, 'val'), transform=validTransform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, log_dir="./logs"):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iter = 0
            for inputs, labels in dataloaders[phase]:
                iter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        if epoch % save_per_epoch == 0:
            net_save_path = os.path.join(log_dir, 'net_params_{}.pkl'.format(epoch))
            torch.save(best_model_wts, net_save_path)

    net_save_path = os.path.join(log_dir, 'best_net_params.pkl')
    torch.save(best_model_wts, net_save_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} Epoch: {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def initialize_pretrained_model(model):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(settings['num_classes'], num_classes)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def run_train():
    # create train log dir
    log_dir = os.path.join('models', ckpt_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 1. create model
    model = pretrainedmodels.__dict__[net_type](num_classes=num_classes, pretrained=None)
    model.last_linear = torch.nn.Linear(512, num_classes)
    initialize_pretrained_model(model)
    print(model)
    print(type(model))

    #     model = nn.DataParallel(model, device_ids=[0]).cuda()  # multi-GPU

    # 2. load data
    train_loader, valid_loader = load_data(model)
    classes_name = train_loader.dataset.classes

    # 3. create optimizer
    params_to_update = model.parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)

    optimizer_ft = optim.SGD(params_to_update, lr=learn_rate, momentum=0.9)

    # 4. setup loss fxn
    criterion = nn.CrossEntropyLoss()

    # 5. train model
    data_loader_dict = {
        "train": train_loader,
        "val": valid_loader
    }
    model_ft, hist = train_model(model, data_loader_dict, criterion, optimizer_ft, num_epochs=train_epoch,
                                 log_dir=log_dir)

    # make confuse matrix
    conf_mat_train, train_acc = validate(model_ft, data_loader_dict["train"], 'train', classes_name)
    conf_mat_valid, valid_acc = validate(model_ft, data_loader_dict["val"], 'valid', classes_name)

    show_confmat(conf_mat_train, classes_name, 'train', log_dir)
    show_confmat(conf_mat_valid, classes_name, 'valid', log_dir)

    # draw pr curve
    if num_classes == 2:
        inference.binary_pr_eval(model_path=os.path.join('models', ckpt_name),
                                 ok_image_path=os.path.join(data_dir, 'val', 'ok'),
                                 ng_image_path=os.path.join(data_dir, 'val', 'ng'),
                                 input_size=input_size,
                                 num_classes=num_classes)


if __name__ == '__main__':
    data_dir = "/root/suye"
    net_type = 'resnet18'  # could be fbresnet152 or inceptionresnetv2
    ckpt_name = "0304_cd_{}".format(net_type)
    num_classes = 2
    batch_size = 32
    input_size = 400
    learn_rate = 0.02
    train_epoch = 200
    save_per_epoch = 50
    settings = {
        'input_space': 'RGB',
        'input_size': [3, input_size, input_size],
        'input_range': [0, 1],
        'mean': [0.53337324, 0.53337324, 0.53337324],
        'std': [0.07127615, 0.07127615, 0.07127615],
        'num_classes': num_classes
    }

    run_train()
