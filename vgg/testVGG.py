"""
Part 1: test the influence of bit-error on original model
"""
import os
import time
import pickle

import copy
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


damage_probability = 0.0001
dataset_root = r'/home/disk/zyr/dataset'
test_number = 5 # 随机试验的次数，多次实验取平均值

print('cuda available:', torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# out是全连接层是输出，不是softmax的输出
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    # print('out range:', torch.min(out).item(), torch.max(out).item())
    # print('out size: ', out.size())
    # print('out isnan rate: ', len(torch.isnan(out).nonzero()) / out.nelement())
    # print('preds: ', preds)
    # print('yb: ', yb)
    return (preds == yb).float().mean().cpu(), len(yb)


def get_binary(value:int)->str:
    arr = []
    for i in range(32):
        arr.append((value >> (31 - i)) & 1)
    return "".join(str(e) for e in arr)


# the number of bits in float
bits = 32


# 返回新的tensor，而不是修改参数tensor
# 以prob的概率出错（翻转比特）
def damage_tensor_bit(tensor, prob):
    masks = torch.zeros_like(tensor, dtype=torch.int32)
    for count in range(bits):
        masks |= (torch.rand_like(tensor) < prob).int()
        masks <<= 1
    # 1 in masks means bit flip, 0 means not change
    return_tensor = (tensor.view(torch.int32) ^ masks).view(torch.float)
    # equals = return_tensor == tensor
    # print('true rate: ', len(equals.nonzero()) / equals.nelement())
    # is_nan = return_tensor != return_tensor
    # print('is_nan rate: ', len(is_nan.nonzero()) / is_nan.nelement())
    return return_tensor


# index代表的就是第几个conv2d或者linear，因为带dropout和不带dropout的层数是不一样的，但是要damage的只有conv2d或者linear层
# 对于features来说，所有层都damage相当于index=[0, 1, 2, 3, 4, 5, 6, 7]，对于classifier来说，是[0, 1, 2]
def damage_layers(pretrained_sequential_model, damage_func, damage_weight_index, damage_bias_index):
    layers = []
    index = -1
    for child_model in pretrained_sequential_model.children():
        if isinstance(child_model, nn.Conv2d) or isinstance(child_model, nn.Linear):
            index += 1
            # print(index)
            if not (index in damage_weight_index or index in damage_bias_index):
                layers += [child_model]
                continue
            copy_model = copy.deepcopy(child_model)
            if index in damage_weight_index:
                # print('original weight:[', torch.min(copy_model.weight), ',', torch.max(copy_model.weight), ']')
                copy_model.weight = nn.Parameter(damage_func(copy_model.weight, damage_probability))
                # print('damaged weight:[', torch.min(copy_model.weight), ',', torch.max(copy_model.weight), ']\n')
            if index in damage_bias_index:
                # print('original bias:[', torch.min(copy_model.bias), ',', torch.max(copy_model.bias), ']')
                copy_model.bias = nn.Parameter(damage_func(copy_model.bias, damage_probability))
                # print('damaged bias:[', torch.min(copy_model.bias), ',', torch.max(copy_model.bias), ']\n')
            layers += [copy_model]
        else:
            layers += [child_model]
    return nn.Sequential(*layers)


class TestDataset(Dataset):
    def __init__(self, test_label, img_dir, transform=None, target_transform=None):
        with open(test_label, 'r') as f:
            self.labels = [int(line[:-1]) for line in f.readlines()]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str('%08d' % (idx + 1)) + '.JPEG')
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def is_grey(PIL_image) -> bool:
    return len(PIL_image.getbands()) == 1


def transform_fun(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if is_grey(image):
        # grey
        # print(filename, 'grey picture')
        rgb = np.stack((image,) * 3, axis=-1)
        return preprocess(Image.fromarray(rgb.astype('uint8')).convert('RGB'))
    else:
        # rgb
        # 有的图像有四个通道，会报这个错误：https://www.it1352.com/2010542.html，所以要先转成RGB
        return preprocess(image.convert('RGB'))


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def preprocess_fun(x, y):
    return x.to(dev), y.to(dev)


class Own_VGG(nn.Module):
    def __init__(self, features, avgpool, classifier):
        super().__init__()
        self.features = features
        self.classifier = classifier
        self.avgpool = avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    bs = 64
    test_ds = TestDataset(os.path.join(dataset_root, 'self_defined_test_label.txt'), os.path.join(dataset_root, 'test'), transform=transform_fun)
    test_dl = WrappedDataLoader(DataLoader(test_ds, batch_size=bs), preprocess_fun)
    print(len(test_ds))
    print('dataset is ready!')

    # model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)  # 第一次下载之后就可以用缓存里的模型文件了
    model.requires_grad_(False)
    model.to(dev)
    model.eval()
    print('model is ready!')

    # test
    #  -------original model--------
    start = time.time()
    with torch.no_grad():
        accuracies, nums = zip(*[accuracy(model(xb), yb) for xb, yb in test_dl])
        acc = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
    print('original model, accuracy:', acc)
    end = time.time()
    print('time: %.2f' % ((end - start) / 60))
    #  --------------damaged model-------------
    features = model.features
    avg_pool = model.avgpool
    classifier = model.classifier
    damage_probabilities = [0, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 1e-7, 1e-8, 1e-9, 1e-10]
    damaged_accuracies = [acc]
    start = time.time()
    for prob in damage_probabilities[1:]:
        print('prob:',prob)
        damage_probability = prob
        damaged_accuracy = []  # the accuracy of damaged model under certain damage_probability
        for i in range(test_number):
            both_damaged_features = damage_layers(features, damage_tensor_bit, [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7])
            both_damaged_classifier = damage_layers(classifier, damage_tensor_bit, [0, 1, 2], [0, 1, 2])
            both_damaged_model = Own_VGG(both_damaged_features, avg_pool, both_damaged_classifier)
            both_damaged_model.eval()
            with torch.no_grad():
                accuracies, nums = zip(*[accuracy(both_damaged_model(xb), yb) for xb, yb in test_dl])
            damaged_accuracy += [np.sum(np.multiply(accuracies, nums)) / np.sum(nums)]
        damaged_accuracies += [np.mean(damaged_accuracy)]
    end = time.time()
    print('time: %.2f' % ((end - start) / 60))
    print('damage_probabilities=', damage_probabilities)
    print('damaged_accuracies=', damaged_accuracies)
