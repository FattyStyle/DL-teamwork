import os
import argparse
from ast import literal_eval

import sys
sys.path.append("..")

import torch
import models
from data import get_dataset
from preprocess import get_transform
from damage import bit_error_tolerance

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        help='dataset name or folder')
    parser.add_argument('--data_path', metavar='DATASET_PATH', default='../../Datasets/',
                        help='dataset main path')
    parser.add_argument('--model', '-a', metavar='MODEL', default='vgg_cifar10_binary',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
    parser.add_argument('--model_config', default='',
                        help='additional architecture configuration')
    parser.add_argument('--input_size', type=int, default=None,
                        help='image input size')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                        help='evaluate model FILE on validation set')
    args = parser.parse_args()

    # model
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}
    if args.model_config != '':
        model_config = dict(model_config, **literal_eval(args.model_config))
    model = model(**model_config)
    if not os.path.isfile(args.evaluate):
        parser.error('invalid checkpoint: {}'.format(args.evaluate))
    checkpoint = torch.load(args.evaluate)
    model.load_state_dict(checkpoint['state_dict'])

    # dataset
    val_transform = get_transform(args.dataset, input_size=args.input_size, augment=False)
    val_data = get_dataset(args.dataset, args.data_path, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    
    model.to('cuda')

    # bit error test
    probs = [1e-2, 1e-4, 1e-6, 1e-8]
    bit_error_tolerance(model, probs, val_loader, 'cuda')
