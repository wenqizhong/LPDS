#coding=utf8

import numpy as np
import os
import torch

import argparse

from data_utils import getTrainVal_loader, getTest_loader
from model.salicon_model import ClassModle


def main():
    parser = argparse.ArgumentParser()
    np.random.seed(12)

    # dataset dir
    parser.add_argument('--train_dataset_dir', type=str, default='', help='dataset') 
    parser.add_argument('--train_stimuli_dir', type=str, default='Images', help='images dir')
    parser.add_argument('--train_scanpath_dir', type=str, default='data_G11-G13', help='scanpath dir')
    parser.add_argument('--test_dataset_dir', type=str, default='/ASD_classification_dataset/Testdata')
    parser.add_argument('--all_dataset_dir', type=str, default='/ASD_classification_dataset/Alldata')
    parser.add_argument('--model_save_path', type=str, default='/model_save/')
    parser.add_argument('--model_name', type=str, default='Salicon_class')
    # train parameters
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training')
    # gpu
    # parser.add_argument('--gpu',default=True,action='store_true')
    parser.add_argument('--train_p', type=str, default='people/p_train_6', help='people use for train')
    parser.add_argument('--test_p', type=str, default='people/p_test_1', help='people use for test')
    parser.add_argument('--gpu', default=True)
    args = parser.parse_args()

    need_people = args.train_p
    dataloaders = getTest_loader(args.train_dataset_dir, args.train_stimuli_dir, args.train_scanpath_dir, test_p=need_people)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = ClassModle()
    model.to(device)
    print('model done')
    test(model, device, dataloaders)


def test(model, device, dataloaders):
    model.load_state_dict(torch.load('checkpoints/model_train.pth'), strict=True)
    model.eval()
    print("Begin test, Device: {}".format(device))

    with torch.no_grad():
        asd_data = {}
        td_data = {}
        td_hard, asd_hard = {}, {}
        for iteration, (img, scanpaths, labels, trans_factor) in enumerate(dataloaders):
            # try:
            img = img.to(device)
            scanpaths = scanpaths.to(device)
            labels = labels.to(device)
            outputs = model(img, scanpaths)
            cse = torch.sum(labels * outputs)
            cse = -cse * torch.log(cse)
            img_name = str(trans_factor[3])[2:-3]   #

            if labels.argmax():  # TD
                if img_name not in td_data:
                    td_data[img_name] = []
                    td_hard[img_name] = cse
                    td_data[img_name].append(outputs)
                else:
                    td_hard[img_name] = td_hard[img_name] + cse
                    td_data[img_name].append(outputs)
            else:
                if img_name not in asd_data:
                    asd_data[img_name] = []
                    asd_hard[img_name] = cse
                    asd_data[img_name].append(outputs)
                else:
                    asd_hard[img_name] = asd_hard[img_name] + cse
                    asd_data[img_name].append(outputs)
        img_entropy = {}
        td_score = {}
        td_kl_all = {}
        for i in td_data:
            td_kl = 0
            td_entropy_all = 0
            for j in td_data[i]:
                td_entropy_all = td_entropy_all + j

            td_entropy_all = td_entropy_all / len(td_data[i])
            td_hard[i] = td_hard[i] / len(td_data[i])
            img_entropy[i] = torch.sum(-td_entropy_all * torch.log(td_entropy_all))
            for j in td_data[i]:
                entropy = torch.sum(-j * torch.log(j))
                kl = torch.sum(-j * torch.log(td_entropy_all)) - entropy
                td_kl = td_kl + kl
            td_kl = td_kl / len(td_data[i])
            td_kl_all[i] = td_kl
            td_score[i] =  td_hard[i] + 0.25*td_kl

        asd_score = {}
        asd_kl_all = {}
        for i in asd_data:
            asd_kl = 0
            asd_entropy_all = 0
            for j in asd_data[i]:
                asd_entropy_all = asd_entropy_all + j
            asd_entropy_all = asd_entropy_all / len(asd_data[i])
            asd_hard[i] = asd_hard[i] / len(asd_data[i])
            img_entropy[i] = torch.sum(-asd_entropy_all * torch.log(asd_entropy_all))
            for j in asd_data[i]:
                entropy = torch.sum(-j * torch.log(j))
                kl = torch.sum(-j * torch.log(asd_entropy_all)) - entropy
                asd_kl = asd_kl + kl
            asd_kl = asd_kl / len(asd_data[i])
            asd_kl_all[i] = asd_kl
            asd_score[i] = asd_hard[i] + 0.25*asd_kl
        img_score = {}
        for i in asd_score:
            img_score[i] = td_score[i] + asd_score[i]
        img_score = sorted(img_score.items(), key=lambda item: item[1])
        score = {}
        for index, i in enumerate(img_score):
            score[i[0]] = img_score[index][1].item()
        score = sorted(score.items(), key=lambda item: item[1])
        outfile = 'score'
        imgfile = 'score_img'
        for (key, hard_simple_p) in score:
            with open(outfile, 'a', encoding='utf-8') as f:
                f.write(str(key) + ':' + str(float(hard_simple_p)) + '\n')
            with open(imgfile, 'a', encoding='utf-8') as f:
                f.write(str(key) + '\n')


if __name__ == '__main__':
    main()
