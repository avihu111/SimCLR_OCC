import numpy as np
from torchvision import datasets


class MySTL10(datasets.STL10):
    def __getitem__(self, item):
        im, lab = super(MySTL10, self).__getitem__(item)
        return im, lab, item


class MyCifar10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(MyCifar10, self).__init__(*args, **kwargs)
        self.labels = np.array(self.targets)

    def __getitem__(self, item):
        im, lab = super(MyCifar10, self).__getitem__(item)
        return im, lab, item


class MyCaltech256(datasets.Caltech256):
    # for generating consistent train test split
    caltech_length = 30600
    perm = np.random.permutation(caltech_length)

    def __init__(self, is_train, *args, **kwargs):
        super(MyCaltech256, self).__init__(*args, **kwargs)
        # only supporting half/half split for now
        self.is_train = is_train
        self.my_indices = MyCaltech256.perm[:MyCaltech256.caltech_length // 2]
        if is_train:
            self.my_indices = MyCaltech256.perm[MyCaltech256.caltech_length // 2:]
        self.labels = np.array(self.y)

    def __getitem__(self, item):
        # indexing on a subset of the permutation
        im, lab = super(MyCaltech256, self).__getitem__(self.my_indices[item])
        return im, lab, item

    def __len__(self):
        return len(self.my_indices)


class MyMNIST(datasets.MNIST):
    def __getitem__(self, item):
        # indexing on a subset of the permutation
        im, lab = super(MyMNIST, self).__getitem__(item)
        return im, lab, item