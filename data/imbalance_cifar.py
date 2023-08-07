from PIL import Image
import os
import os.path
import copy
import errno
import numpy as np
import sys
import pickle
import random
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.transforms as transforms


def get_img_num_per_cls(dataset, imb_factor=None, num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = (50000 - num_meta) / 10
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000 - num_meta) / 100
        cls_num = 100

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


# def uniform_mix_C(mixing_ratio, num_classes, num_list):
#     '''
#     returns a linear interpolation of a uniform matrix and an identity matrix
#     '''
#     num_cls = np.array(num_list)
#     C = np.zeros((num_classes, num_classes))
#     for i in range(num_classes):
#         C[i] = mixing_ratio * num_cls / (num_cls.sum() - num_cls[i])
#         C[i, i] = 1 - mixing_ratio
#     return C

def uniform_mix_C(mixing_ratio, num_classes, num_list):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    # return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
    #     (1 - mixing_ratio) * np.eye(num_classes)
    num_cls = np.array(num_list)
    C = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        C[i] = mixing_ratio * num_cls / (num_cls.sum())
        C[i, i] = 1 - mixing_ratio + mixing_ratio * num_cls[i] / (num_cls.sum())
    return C


class CIFAR10_im(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    augment_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    easy_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = 'cifar10'

    def __init__(self, root='', train=True, meta=True, num_meta=1000,
                 corruption_prob=0, corruption_type='unif', transform='easy', target_transform=None,
                 download=True, seed=1, imblance=False, imb_factor=None):
        self.root = root
        if transform == 'easy':
            self.transform = self.easy_transform
        elif transform == 'hard':
            self.transform = self.augment_transform
        self.target_transform = target_transform


        self.train = train  # training set or test set
        self.meta = meta
        self.corruption_prob = corruption_prob
        self.num_meta = num_meta
        self.imblance = imblance
        self.imb_factor = imb_factor
        np.random.seed(seed)
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_coarse_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                    img_num_list = [int(self.num_meta / 10)] * 10
                    num_classes = 10
                else:
                    self.train_labels += entry['fine_labels']
                    self.train_coarse_labels += entry['coarse_labels']
                    img_num_list = [int(self.num_meta / 100)] * 100
                    num_classes = 100
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            data_list_val = {}
            for j in range(num_classes):
                data_list_val[j] = [i for i, label in enumerate(self.train_labels) if label == j]

            idx_to_meta = []
            idx_to_train_list = {}
            print(img_num_list)

            for cls_idx, img_id_list in data_list_val.items():
                np.random.shuffle(img_id_list)
                img_num = img_num_list[int(cls_idx)]
                idx_to_meta.extend(img_id_list[:img_num])
                idx_to_train_list[cls_idx] = img_id_list[img_num:]

            if self.imblance is True:
                img_imblance_num_list = get_img_num_per_cls(self.dataset, self.imb_factor, self.num_meta)
                for cls_idx, img_id_list in idx_to_train_list.items():
                    random.shuffle(img_id_list)
                    img_num = img_imblance_num_list[int(cls_idx)]
                    idx_to_train_list[cls_idx] = img_id_list[:img_num]
                print('imb_data: ', img_imblance_num_list)

            idx_to_train = []
            for cls_idx, img_id_list in idx_to_train_list.items():
                idx_to_train.extend(img_id_list)

            if meta is True:
                self.train_data = self.train_data[idx_to_meta]
                self.train_labels = list(np.array(self.train_labels)[idx_to_meta])
                self.true_labels = copy.deepcopy(self.train_labels)
            else:
                self.train_data = self.train_data[idx_to_train]
                self.train_labels = list(np.array(self.train_labels)[idx_to_train])
                self.true_labels = copy.deepcopy(self.train_labels)

                if corruption_type == 'unif':
                    C = uniform_mix_C(self.corruption_prob, num_classes, img_imblance_num_list)
                    self.C = C
                else:
                    self.C = np.eye(num_classes)
                print(self.C)

                for i in range(len(self.train_labels)):
                    self.train_labels[i] = np.random.choice(num_classes, p=self.C[self.train_labels[i]])
                a=0
                for i in range(len(self.train_labels)):
                    if self.train_labels[i] == self.true_labels[i]:
                        a+=1
                self.noise_or_not=[int(self.train_labels[i] == self.true_labels[i]) for i in range(len(self.true_labels))]
                # print(len(self.noise_or_not))
                print("ture noise rate is "+str(a/len(self.train_labels)))
        else:
            f = self.test_list[0][0]
            file = os.path.join(root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            img, target, true_target, noise_or_not = self.train_data[index], self.train_labels[index], self.true_labels[index], self.noise_or_not[index]
        else:
            img, target, true_target, noise_or_not = self.test_data[index], self.test_labels[index], self.test_labels[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            true_target = self.target_transform(true_target)

        return img, target, index, true_target
        # return img, target, index, true_target
        # return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return 10000

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


class CIFAR100_im(CIFAR10_im):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    dataset = 'cifar100'
