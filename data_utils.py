import numpy as np
import math
import os
import sys
import torch
# import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import scipy.io as scio

vgg16_mean=np.array([123.68,116.779,103.939])/255.
vgg16_std=np.array([0.229,0.224,0.225])


def split_scanpaths(data):
    starts = np.where(data['idx'] == 0)[0]
    ends = np.append(starts[1:], len(data))
    assert starts.shape == ends.shape

    return [data[start:end][['x', 'y', 'duration']] for start, end in zip(starts, ends)]



class DiscrimiDataset(Dataset):
    """
    ASDdataset : Discrimility ASD dataset
    Dataset wrapping images, scanpaths and label(ASD:'1' or TD:'0').

    dataset_root
        |____ scanpath
              |____asd_scanpath.mat
              |____td_scanpath.mat
        |____ stimuli
              |____image.jpg
     """

    def __init__(self, dataset_root, stimuli_dir, scanpath_dir, need_people=None, need_images=None, transform=None,
                 use_cache=False, type='train'):
        super(DiscrimiDataset, self).__init__()
        print('creating dataset...')

        self.dataset_root, self.stimuli_dir, self.scanpath_dir = dataset_root, stimuli_dir, scanpath_dir

        self.scanpaths = []
        self.labels = []

        if os.path.basename(dataset_root) == 'our_scanpath':
            self.get_scanpath(need_p=need_people, need_img=need_images)
        elif os.path.basename(dataset_root) == 'gender':
            self.get_gender_data(need_p=need_people, need_img=need_images)
        else:
            self.get_scanpath_data(need_p=need_people, need_img=need_images)
        if os.path.basename(dataset_root) == 'age':
            for file in self.scanpaths:
                if '18mos' in file:
                # if file[0:3] == 'asd':
                    self.labels.append([1.0, 0.0])
                elif '30mos' in file:
                    self.labels.append([0.0, 1.0])
                else:
                    raise
        elif os.path.basename(dataset_root) == 'task': 
            if need_people.split('/', 2)[1] == 'free_obj':    
                for file in self.scanpaths:
                    if 'freeview' in file:
                    # if file[0:3] == 'asd':
                        self.labels.append([1.0, 0.0])
                    elif 'objsearch' in file:
                        self.labels.append([0.0, 1.0])
            elif need_people.split('/', 2)[1] == 'free_sal':
                    if 'freeview' in file:
                    # if file[0:3] == 'asd':
                        self.labels.append([1.0, 0.0])
                    elif 'salview' in file:
                        self.labels.append([0.0, 1.0])
                    else:
                        raise
        elif os.path.basename(dataset_root) == 'gender':
            for file in self.scanpaths:
                label = file.split('_', 1)[0]
                if label == '男':
                    self.labels.append([1.0, 0.0])
                elif label == '女':
                    self.labels.append([0.0, 1.0])
                else:
                    print(label)
                    raise
        else:
            for file in self.scanpaths:
                if 'ASD' in file:
                # if file[0:3] == 'asd':
                    self.labels.append([1.0, 0.0])
                elif '正常' in file or 'TD' in file:
                    self.labels.append([0.0, 1.0])
                else:
                    raise

        self.transform = transform
        self.use_cache = use_cache
        self.lens = len(self.scanpaths)  # scanpaths_len

    def get_scanpath(self, need_p=None, need_img=None):
        if need_p is None and need_img is None:
            print('the need people and the need images is None')
            [self.scanpaths.append(f) for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir))]

        elif need_p is not None and need_img is None:
            print('the need images is None\n the need people is {}'.format(str(need_p)))
            need_p = np.genfromtxt(os.path.join(self.dataset_root, need_p), delimiter=',', 
                                dtype=str, case_sensitive='lower')
            for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir)):
                p_name = f.split(' ', 2)[1]
                if p_name in need_p:
                    self.scanpaths.append(f)

        elif need_img is not None and need_p is not None:
            print('the need images is {}\n the need people is {}'.format(need_img, need_p))
            need_img = np.genfromtxt(need_img, delimiter=',', 
                                dtype=str, case_sensitive='lower')
            need_p = np.genfromtxt(os.path.join(self.dataset_root, need_p), delimiter=',', 
                                dtype=str, case_sensitive='lower')
            for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir)):
                p_name = f.split(' ', 2)[1]
                img_name = f.split('_', 1)[1][:-4]
                if 'ASD' in img_name or '正常' in img_name:
                    img_name = img_name.split('_', 1)[1]
                if p_name in need_p and img_name in need_img:
                    self.scanpaths.append(f)

        else:
            raise


    def get_scanpath_data(self, need_p=None, need_img=None):
        if need_p is None:
            print('the need people and the need images is None')
            [self.scanpaths.append(f) for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir))]

        elif need_img is not None and need_p is not None:
            print('the need images is {}\n the need people is {}'.format(str(need_img), str(need_p)))
            need_p = np.genfromtxt(os.path.join(self.dataset_root, need_p), delimiter=',', 
                                dtype=str, case_sensitive='lower')
            need_img = np.genfromtxt(need_img, dtype=str, delimiter=',', case_sensitive='lower')
            for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir)):
                p_name = f.split('_', 2)[2][:-4]
                i_name = f.split('_', 2)[1]
                if p_name in need_p and i_name in need_img:
                    self.scanpaths.append(f)
        
        else:
            print('the need images is None\n the need people is {}'.format(str(need_p)))
            need_p = np.genfromtxt(os.path.join(self.dataset_root, need_p), delimiter=',', 
                                dtype=str, case_sensitive='lower')
            for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir)):
                p_name = f.split('_', 2)[2][:-4]
                if p_name in need_p:
                    self.scanpaths.append(f)

    def get_gender_data(self, need_p=None, need_img=None):
        if need_p is None:
            print('the need people is none')
            [self.scanpaths.append(f) for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir))]

        elif need_img is not None and need_p is not None:
            print('the need images is {}\n the need people is {}'.format(str(need_img), str(need_p)))
            need_p = np.genfromtxt(os.path.join(self.dataset_root, need_p), delimiter=',', 
                                dtype=str, case_sensitive='lower')
            need_img = np.genfromtxt(need_img, dtype=str, delimiter=',', case_sensitive='lower')
            for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir)):
                p_name = f.split('_')[-1][:-4]
                i_name = f.split('_', 1)[1][:-4]
                i_name = i_name.rsplit('_', 1)[0]
                if p_name in need_p and i_name in need_img:
                    self.scanpaths.append(f)
        else:
            print('the need images is None\n the need people is {}'.format(str(need_p)))
            need_p = np.genfromtxt(os.path.join(self.dataset_root, need_p), delimiter=',', 
                                dtype=str, case_sensitive='lower')
            for f in os.listdir(os.path.join(self.dataset_root, self.scanpath_dir)):
                p_name = f.split('_')[-1][:-4]
                if p_name in need_p:
                    self.scanpaths.append(f)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, fixmap) where map is the fixation map of the image.
        """

        scanpaths_dir = os.path.join(self.dataset_root, self.scanpath_dir,  self.scanpaths[index])

        scanpaths = scio.loadmat(scanpaths_dir)     # scanpaths:(image_name, fixation_order, x, y, duration)
        scanpaths = scanpaths['save_data']
        imgfn = scanpaths[0, 0]    # imagefilename
        # imgfn = imgfn.replace(' ', '')
        if len(scanpaths[0]) >= 5:
            duration = list(map(float, scanpaths[:, 4]))
        else:
            duration = None

        imgfn = imgfn.replace(' ', '')
        image_dir = os.path.join(self.dataset_root, self.stimuli_dir, str(imgfn) + '.jpg')


        try:
            image = Image.open(image_dir)
            # plt.imshow(image)
            width, height = image.size
        except FileNotFoundError:
            print('=======')
            print(scanpaths_dir)
            print(imgfn)
            print(image_dir)
            print('=======')
            raise
        x_scanpaths, y_scanpaths = list(map(float, scanpaths[:, 2])), list(map(float, scanpaths[:, 3]))
        x_scanpaths, y_scanpaths = list(map(int, x_scanpaths)), list(map(int, y_scanpaths))
        x_scanpaths, y_scanpaths = np.array(x_scanpaths), np.array(y_scanpaths)
        x_scanpaths, y_scanpaths = np.clip(x_scanpaths, 0, 1920), np.clip(y_scanpaths, 0, 1080)
        

        x_trans_factor = width / 16        # test
        y_trans_factor = height / 16
        scanpaths = [np.floor((x_scanpaths - 1) / x_trans_factor), np.floor((y_scanpaths - 1) / y_trans_factor)]

        scanpaths = torch.transpose(torch.from_numpy(np.stack(scanpaths, axis=0)), 0, 1)  # scanpaths: N*2 tensor

        duration = torch.from_numpy(np.array(duration))

        if self.transform is None:
            img = image
        elif all([x in self.transform.keys() for x in ['fine', 'ori', 'global']]):
            img = self.transform['fine'](image)
            # globalImg = self.transform['global'](image)
        else:
            raise NotImplemented

        label = torch.tensor(self.labels[index])

        return img, scanpaths, label, [x_trans_factor, y_trans_factor, self.scanpaths[index], imgfn, duration]

    def __len__(self):
        return self.lens



def getTrainVal_loader(train_dataset_dir, img_dir, scanpath_dir, train_p=None, need_images=None, batch_s=None, shuffle=True, val_split=0.1):
    vgg16_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) / 255.
    vgg16_std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'fine': transforms.Compose([
            transforms.Resize((256, 256), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'ori': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'global': transforms.Compose([
            transforms.Resize((65, 65), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ])
    }
    print('...')
    trainval_dataset = DiscrimiDataset(train_dataset_dir, img_dir, scanpath_dir, need_people=train_p, need_images=need_images, transform=data_transforms)

    dataset_size = len(trainval_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(trainval_dataset, batch_size=batch_s, sampler=train_sampler, num_workers=6)
    val_loader = DataLoader(trainval_dataset, batch_size=batch_s, sampler=valid_sampler, num_workers=6)

    trainval_loaders = {'train': train_loader, 'val': val_loader}

    return trainval_loaders


def getTest_loader(test_dataset_dir, img_dir, scanpath_dir, test_p=None, need_images=None, shuffle=True):
    vgg16_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) / 255.
    vgg16_std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'fine': transforms.Compose([
            transforms.Resize((256, 256), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'ori': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'global': transforms.Compose([
            transforms.Resize((65, 65), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ])
    }

    test_dataset = DiscrimiDataset(test_dataset_dir, img_dir, scanpath_dir, need_people=test_p, need_images=need_images, transform=data_transforms)

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=6)

    return test_loader

