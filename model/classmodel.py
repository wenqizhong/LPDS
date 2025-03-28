import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence, pack_sequence
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


class ClassModle(nn.Module):
    def __init__(self):
        super(ClassModle, self).__init__()
        print('Making model...')
   
        self.features = models.vgg16(pretrained=True).features[0:30]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096), #4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
            nn.Softmax(),
        )


    def forward(self, x, scanpath, duration=None):
        '''
        x: B*C*H*W
        scanpath: B*N*2    'N' denotes the length of scanpath '2'denotes 'x' and 'y'
        scanpath: N*2    'N' denotes the length of scanpath '2'denotes 'x' and 'y'
        [x, y] or [y, x]
        print(scanpath.size())      torch.Size([1, 6, 2])

        label: [ASD, TD]
        '''
        x = self.features(x)
        x = x[:, :, scanpath[:, :, 1].long(), scanpath[:, :, 0].long()]           # note the position of [:,0] and [:,1]
        # x = self._select_feat(x, scanpath)
        # print(x.size()) : B*C*1*N
        x = self.avg_pool(x)
        x = x.view(x.size(0), 512)
        x = self.classifier(x)
        return x

    def _select_feat(self, x, scanpath):
        for i in range(scanpath.dim(1)):
            x_list = x[:, :, scanpath[:, i, 1], scanpath[:, i, 0]]
        x = torch.stack(x_list, dim=2)
        return x
