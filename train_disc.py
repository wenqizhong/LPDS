#coding=utf8
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

import time
import argparse

from data_utils import getTrainVal_loader, getTest_loader
from model.classmodel import ClassModle
import trainer


def main():
    parser = argparse.ArgumentParser()
    np.random.seed(12)


    parser.add_argument('--train_dataset_dir', type=str, default='/dataset/saliency4asd')
    parser.add_argument('--train_stimuli_dir', type=str, default='Images', help='images dir')
    parser.add_argument('--train_scanpath_dir', type=str, default='train', help='scanpath dir')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/saliency4asd', help='the path to save model')
    parser.add_argument('--logs', type=str, default='logs/saliency4asd/scanformer', help='the path to save training logs')
    parser.add_argument('--model_name', type=str, default='scanformer')

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training, only supports 1')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for training')
    parser.add_argument('--decay', type=float, default=0.0005, help='weight decay for training')
    parser.add_argument('--epochs', type=int, default=20, help='epoch for training')


    parser.add_argument('--train_p', type=str, default='people/p_train_1', help='people use for train')
    parser.add_argument('--test_p', type=str, default='people/p_test_1', help='people use for test')
    parser.add_argument('--gpu', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    # need_img = 'score-fisher_gender30p/fisher_ampltude_img'
    # need_img = None
    dataloaders = getTrainVal_loader(args.train_dataset_dir, args.train_stimuli_dir, args.train_scanpath_dir, train_p=args.train_p, need_images=need_img, batch_s=args.batch_size)

    if not os.path.exists("%s/path-%s" %(args.checkpoint, args.model_name)):
        os.mkdir("%s/path-%s" %(args.checkpoint, args.model_name))

    model = ClassModle()
    
    print('model done')
    model.to(device)
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay,
                             nesterov=True)

    criterion = torch.nn.BCELoss()
    print("Begin train, Device: {}".format(device))

    trainer.train_model(args, model, device, dataloaders, criterion, optimizer_ft, num_epochs=args.epochs,
                check_point=1)


if __name__ == '__main__':
    main()