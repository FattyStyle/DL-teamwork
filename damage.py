import copy

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn


# 以prob的概率让tensor中的每一个单元出错
def damage_tensor(tensor, prob):
    masks = torch.zeros_like(tensor, dtype=torch.int32)
    for count in range(32):
        masks |= (torch.rand_like(tensor) < prob).int()
        masks <<= 1
    # 1 in masks means bit-flip, 0 means not change
    return (tensor.view(torch.int32) ^ masks).view(torch.float)


def damage_model(original_model, prob):
    copy_model = copy.deepcopy(original_model)
    for name, param in copy_model.named_parameters():
        # print(name)
        param.data = damage_tensor(param.data, prob)
    return copy_model


def test(model, dataset_loader, criterion, device):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataset_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item() * targets.size(0)
            total += targets.size(0)
            _, predictions = torch.max(outputs, 1)
            correct += torch.sum(predictions == targets.data)
    return loss / total, correct.double() / total


def bit_error_tolerance(model, prob, dataset_loader, device):
    criterion = nn.CrossEntropyLoss()
    loss, acc = test(model, dataset_loader, criterion, device)
    print("original model ==> loss:{%.3f}, acc:{%.3f}" % (loss, acc))
    model = damage_model(model, prob)
    loss, acc = test(model, dataset_loader, criterion, device)
    print("damaged model ==> loss:{%.3f}, acc:{%.3f}" % (loss, acc))


if __name__ == '__main__':
    # 这个文件的代码用于模拟RRAM上的比特错误情形，下面是使用示例
    print('==> Building model..')
    # 随便找的一个在cifar上预训练好的模型 https://github.com/chenyaofo/pytorch-cifar-models
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 下载好数据之后记得将download改为False
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    bit_error_tolerance(model, 0.01, test_loader, device)