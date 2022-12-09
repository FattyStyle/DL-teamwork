from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.hub
import numpy as np
import time
import pickle
from torchvision.datasets import ImageNet, ImageFolder
from pytorchtools import EarlyStopping


print('cuda_available: ', torch.cuda.is_available())
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

small = 32

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M',
          512, 512, 'M', 512, 512, 'M'],

    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
          512, 512, 'M', 512, 512, 'M'],

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
          512, 512, 512, 'M', 512, 512, 512, 'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
          'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

    'S': [4, 'M', 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M']
}


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
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return x.to(dev), y.to(dev)


def get_data(train_ds, valid_ds, bs):
    return (
        WrappedDataLoader(DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4), preprocess_fun),
        WrappedDataLoader(DataLoader(valid_ds, batch_size=bs), preprocess_fun),
    )


def loss_batch(model, loss_func, xb, yb, opt=None, scheduler=None):
    loss = loss_func(model(xb), yb)
    # print(loss)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    if scheduler is not None:
        scheduler.step()

    return loss.item(), len(xb)


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    # print('out: ', out)
    # print('preds: ', preds)
    # print('yb: ', yb)
    return (preds == yb).float().mean().cpu()


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, scheduler):
    print('time: ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
    train_loss_arr = np.zeros(epochs)
    valid_loss_arr = np.zeros(epochs)
    accuracy_arr = np.zeros(epochs)
    train_accuracy_arr = np.zeros(epochs)
    early_stopping = EarlyStopping(patience=5, verbose=True)

    model.eval()  # 让model变成测试模式，对dropout和batch normalization的操作在训练和测试的时候是不一样的
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
        )
        # train_losses, train_nums = zip(
        #     *[loss_batch(model, loss_func, xb, yb) for xb, yb in train_dl]
        # )
        accuracies = [accuracy(model(xb), yb) for xb, yb in valid_dl]

    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    print('initial validation loss:', val_loss,'accuracy:', np.sum(np.multiply(accuracies, nums)) / np.sum(nums), 'time: ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
    for epoch in range(epochs):
        print('epoch',epoch)
        model.train()
        # for xb, yb in train_dl:
        #     loss_batch(model, loss_func, xb, yb, opt)
        train_losses, train_nums = zip(
            *[loss_batch(model, loss_func, xb, yb, opt, scheduler) for xb, yb in train_dl]
        )

        model.eval()  # 让model变成测试模式，对dropout和batch normalization的操作在训练和测试的时候是不一样的
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
            # train_losses, train_nums = zip(
            #     *[loss_batch(model, loss_func, xb, yb) for xb, yb in train_dl]
            # )
            accuracies = [accuracy(model(xb), yb) for xb, yb in valid_dl]
            train_acc = [accuracy(model(xb), yb) for xb, yb in train_dl]

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        # print(epoch, 'validation loss:', val_loss, 'train loss:', train_loss, 'time: ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        train_loss_arr[epoch] = train_loss
        valid_loss_arr[epoch] = val_loss
        train_accuracy_arr[epoch] = np.sum(np.multiply(train_acc, train_nums)) / np.sum(train_nums)
        accuracy_arr[epoch] = np.sum(np.multiply(accuracies, nums)) / np.sum(nums)
        print(epoch, 'validation loss:', val_loss, 'train loss:', train_loss, 'valid accuracy:', accuracy_arr[epoch],
              'train accuracy: ', train_accuracy_arr[epoch], 'time: ', time.strftime("%m-%d %H:%M:%S", time.localtime()))
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_with_dropout(cfg, p=0.5):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.Dropout(p), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_with_variable_dropout(cfg, p):
    layers = []
    in_channels = 3
    i = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.Dropout(p[i]), nn.ReLU(inplace=True)]
            in_channels = v
            i += 1
    return nn.Sequential(*layers)


def make_classifier():
    return nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1000)
        # nn.Linear(4096, 10) # 10 type for smaller model
    )


class Own_VGG(nn.Module):
    def __init__(self, features, avgpool, classifier):
        super().__init__()
        self.features = features
        self.classifier = classifier
        # self.adapter = nn.AdaptiveAvgPool2d(7)
        self.avgpool = avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # 有了这一步就不需要flatten层了
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = ImageNet(root=r'/home/disk/zyr/dataset', split='train', transform=preprocess)
    valid_ds = ImageNet(root=r'/home/disk/zyr/dataset', split='val', transform=preprocess)
    print(len(train_ds), len(valid_ds))
    bs = 256
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    print('datasets are ready!')

    model = Own_VGG(make_layers_with_dropout(cfg['A']), nn.AdaptiveAvgPool2d(7), make_classifier())
    model.to(dev)
    print('model is ready!')

    epochs = 1000
    opt = optim.Adam(model.parameters(), lr=0.001)
    # opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    # scheduler = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    scheduler = None
    loss_func = F.cross_entropy
    start = time.time()
    fit(epochs, model, loss_func, opt, train_dl, valid_dl, scheduler)
    end = time.time()
    print('time', (end - start) / 60, 'min')
    with open("vgg_model.pkl", "wb") as f:
        pickle.dump(model, f, -1)