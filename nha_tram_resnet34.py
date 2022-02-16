#!/usr/bin/env python
# coding: utf-8

# In[16]:


import torch
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import pandas

from os.path import join
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from torch_deform_conv.layers import ConvOffset2D


# In[2]:


lr = 0.001
num_epochs = 500
batch_size = 64
criterion = nn.CrossEntropyLoss()
#step_size = 3
#gamma = 0.1

IMG_SIZE = (128, 128)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

data_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])


# In[3]:


#train_path = '/home/vietnh41-vtnet-u/Documents/Project/dogs_vs_cats/train/'
#test_path = '/home/vietnh41-vtnet-u/Documents/Project/dogs_vs_cats/test1/'

ori_path = '/Disk01/personal_folders/haint126/dataset/cleaned_classifier_v4/old_cleaned_classifier'
ldir = os.listdir(ori_path)
nha_tram_dict = []
for class_name in ldir:
    nha_tram_dict.append((class_name, len(os.listdir(join(ori_path, class_name)))))


# In[4]:


train_path = '/home/sohoa1/vietnh41/nha_tram_classify/train'
val_path = '/home/sohoa1/vietnh41/nha_tram_classify/val'

train_dir = os.listdir(train_path)
val_dir = os.listdir(val_path)


# In[5]:


mong_cot_vuong_path = os.listdir(train_path + '/' + train_dir[0])
mt_nha_tram_path = os.listdir(train_path + '/' + train_dir[1])
mong_co_path = os.listdir(train_path + '/' + train_dir[2])
dinh_cot_vuong_path = os.listdir(train_path + '/' + train_dir[3])


# In[6]:


val_mong_cot_vuong_path = os.listdir(val_path + '/' + val_dir[0])
val_mt_nha_tram_path = os.listdir(val_path + '/' + val_dir[1])
val_mong_co_path = os.listdir(val_path + '/' + val_dir[2])
val_dinh_cot_vuong_path = os.listdir(val_path + '/' + val_dir[3])


# In[7]:


class dataset(Dataset):
    def __init__(self, file_list, path, label, target, transform=None):
        self.file_list = file_list
        self.path = path
        self.label = label
        self.target = target
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(join(self.path, self.file_list[idx]))
        
        if self.transform:
            img = self.transform(img)
        img = img.numpy()
        return img.astype('float32'), self.target


# In[8]:


mcv_train = dataset(mong_cot_vuong_path, train_path + '/' + train_dir[0], train_dir[0], 0, data_transform)
mtnt_train = dataset(mt_nha_tram_path, train_path + '/' + train_dir[1], train_dir[1], 1, data_transform)
mc_train = dataset(mong_co_path, train_path + '/' + train_dir[2], train_dir[2], 2, data_transform)
dcv_train = dataset(dinh_cot_vuong_path, train_path + '/' + train_dir[3], train_dir[3], 3, data_transform)

traindata = ConcatDataset([mcv_train, mtnt_train, mc_train, dcv_train])

train_dataloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)


# In[9]:


mcv_val = dataset(val_mong_cot_vuong_path, val_path + '/' + val_dir[0], val_dir[0], 0, data_transform)
mtnt_val = dataset(val_mt_nha_tram_path, val_path + '/' + val_dir[1], val_dir[1], 1, data_transform)
mc_val = dataset(val_mong_co_path, val_path + '/' + val_dir[2], val_dir[2], 2, data_transform)
dcv_val = dataset(val_dinh_cot_vuong_path, val_path + '/' + val_dir[3], val_dir[3], 3, data_transform)

valdata = ConcatDataset([mcv_val, mtnt_val, mc_val, dcv_val])

val_dataloader = DataLoader(valdata, batch_size=batch_size, shuffle=False, num_workers=2)


# In[10]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[14]:


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, deform=False, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if deform:
            self.offset1 = ConvOffset2D(inplanes)
        else:
            self.offset1 = None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if deform:
            self.offset2 = ConvOffset2D(planes)
        else:
            self.offset2 = None
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        if self.offset1:
            out = self.offset1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.offset2:
            out = self.offset2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, deform=False, num_classes=4, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], deform, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, deform=False, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, deform, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, deform, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, deform, pretrained, progress, **kwargs):
    model = ResNet(block, layers, deform, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet34(deform=False, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], deform, pretrained, progress, **kwargs)


# In[17]:


model = resnet34(deform=False)


# In[18]:


def evaluate_accuracy(dataloader, model, device=device):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in dataloader:
        # If device is the GPU, copy the data to the GPU.
        X, y = X.to(device), y.to(device)
        model.eval()
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(model(X), dim=1) == y))
            n += y.shape[0]
    return acc_sum.item()/n


# In[19]:


def train_ch5(model, train_iter, test_iter, criterion, num_epochs, device, lr=None):
    """Train and evaluate a model with CPU or GPU."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    
    model.apply(init_weights)
    if torch.cuda.is_available():
        print('training on', device)
        model.to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    val_acc = []
    best_acc = 0.0
    model_path = '/home/sohoa1/vietnh41/'
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        train_acc_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        n, start = 0, time.time()
        for X, y in train_iter:
            model.train()
            
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device) 
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, model, device)
        train_loss_epoch = train_l_sum/n
        train_acc_epoch = train_acc_sum/n
        train_loss.append(train_loss_epoch.item())
        train_acc.append(train_acc_epoch.item())
        val_acc.append(test_acc)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_loss_epoch, train_acc_epoch, test_acc,
                 time.time() - start))
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path + 'nha_tram_resnet34500.pt')
    return train_loss, train_acc, val_acc, best_acc


# In[ ]:


train_loss, train_acc, val_acc, best_acc = train_ch5(model, train_dataloader, val_dataloader, criterion, num_epochs, device, lr=lr)

df = pandas.DataFrame(data={"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
df.to_csv("/home/sohoa1/vietnh41/nha_tram_resnet34500.csv", sep=',',index=False)

print(best_acc)



