# -*- coding: utf-8 -*-
"""
# Resnet Implementation
# Authors: suye
# Date: 2020/03/26
"""
from tqdm import tqdm
import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data

from resnet import ResNet

train_data = torchvision.datasets.MNIST(
    './mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.MNIST(
    './mnist', train=False, transform=torchvision.transforms.ToTensor()
)


def prepare_data_loader():
    print("train_data:", train_data.train_data.size())
    print("train_labels:", train_data.train_labels.size())
    print("test_data:", test_data.test_data.size())

    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=64)

    return train_loader, test_loader


def main():
    num_classes = 10
    model = ResNet(depth=50, num_classes=num_classes, input_c=1)
    print(model)

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    train_loader, test_loader = prepare_data_loader()
    for epoch in range(10):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in tqdm(test_loader):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.data.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            train_data)), train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in tqdm(test_loader):
            batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_data)), eval_acc / (len(test_data))))


if __name__ == '__main__':
    main()
