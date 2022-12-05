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
    return loss / total, correct.double().item() / total


class ClipModel(nn.Module):
    def __init__(self, inner_model, min, max):
        super(ClipModel, self).__init__()
        self.model = inner_model
        self.min = min
        self.max = max

    def forward(self, x):
        for param in self.model.parameters():
            param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
            param.data = torch.clamp(param.data, max=self.max, min=self.min)
        return self.model(x)


def bit_error_tolerance(model, probs, dataset_loader, device):
    criterion = nn.CrossEntropyLoss()
    loss, acc = test(model, dataset_loader, criterion, device)
    print("original model ==> loss:{%.3f}, acc:{%.3f}" % (loss, acc))
    print("damaged model ==> ")
    for prob in probs:
        print("damage probability: ", prob)
        damaged_model = damage_model(model, prob)
        damaged_loss, damaged_acc = test(damaged_model, dataset_loader, criterion, device)
        print("damaged loss:{%.3f}, damaged acc:{%.3f}, loss rate:{%.3f}, acc rate:{%.3f}"
              % (damaged_loss, damaged_acc, damaged_loss / loss, damaged_acc / acc))
        clipped_model = ClipModel(damaged_model, -1.0, 1.0)
        clipped_loss, clipped_acc = test(clipped_model, dataset_loader, criterion, device)
        print("clipped loss:{%.3f}, clipped acc:{%.3f}, loss rate:{%.3f}, acc rate:{%.3f}"
              % (clipped_loss, clipped_acc, clipped_loss / loss, clipped_acc / acc))


if __name__ == '__main__':
    # 这个文件的代码用于模拟RRAM上的比特错误情形，下面是使用示例
    # prepare model
    # 随便找的一个在cifar上预训练好的模型 https://github.com/chenyaofo/pytorch-cifar-models
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    # prepare dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 下载好数据之后记得将download改为False
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    probs = [1e-2, 1e-4, 1e-6, 1e-8]
    bit_error_tolerance(model, probs, test_loader, device)