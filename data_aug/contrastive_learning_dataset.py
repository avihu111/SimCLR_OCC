import numpy as np
import torch
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.my_datasets import MySTL10, MyCifar10, MyCaltech256
from data_aug.view_generator import ContrastiveLearningViewGenerator, FixImageChannels
from torch.utils.data import Dataset


class ContrastiveLearningDataset:
    def __init__(self, root_folder, batch_size, workers, relevant_classes, num_labeled_examples, name):
        self.root_folder = root_folder
        self.batch_size = batch_size
        self.workers = workers
        self.num_labeled_examples = num_labeled_examples
        self.relevant_classes = relevant_classes
        self.name = name
        self.init_datasets(name)

        all_labeled_indices = np.isin(self.train_labeled_ds.labels, self.relevant_classes).nonzero()[0]
        if self.name == 'caltech256':
            all_labeled_indices = \
                np.isin(self.train_labeled_ds.labels[self.train_labeled_ds.my_indices],
                        self.relevant_classes).nonzero()[0]
        assert (self.num_labeled_examples % 4) == 0 and self.num_labeled_examples <= len(all_labeled_indices)
        self.positive_indices = np.random.permutation(all_labeled_indices)[:self.num_labeled_examples]

    def init_datasets(self, name):
        if name == 'cifar10':
            train_transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(32), 2)
            self.train_ds = MyCifar10(self.root_folder, train=True,
                                      transform=train_transform, download=True)
            test_transform = transforms.ToTensor()
            self.test_ds = MyCifar10(self.root_folder, train=False, transform=test_transform,
                                     download=True)
            self.train_labeled_ds = MyCifar10(self.root_folder, train=True, transform=test_transform,
                                              download=True)

        elif name == 'stl10':
            train_transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(96), 2)
            self.train_ds = MySTL10(self.root_folder, split='train+unlabeled', transform=train_transform, download=True)
            test_transform = transforms.ToTensor()
            self.test_ds = MySTL10(self.root_folder, split='test', transform=test_transform, download=True)
            self.train_labeled_ds = MySTL10(self.root_folder, split='train', transform=test_transform,
                                            download=True)

        elif name == 'caltech256':
            simclr_transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(224), 2)
            data_transforms = transforms.Compose([transforms.Resize((224, 224)), FixImageChannels(), simclr_transform])
            self.train_ds = MyCaltech256(is_train=True, root=self.root_folder, transform=data_transforms, download=True)
            test_transform = transforms.Compose(
                [transforms.Resize((224, 224)), FixImageChannels(), transforms.ToTensor()])
            self.test_ds = MyCaltech256(is_train=False, root=self.root_folder, transform=test_transform, download=True)
            self.train_labeled_ds = MyCaltech256(is_train=True, root=self.root_folder, transform=test_transform,
                                                 download=True)
        else:
            raise NotImplementedError(f'{name} is not supported')

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_augmented_dataset(self, labeled_boost=3):
        weights = np.ones(len(self.train_ds))
        weights[self.positive_indices] = labeled_boost
        sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights))
        return torch.utils.data.DataLoader(self.train_ds, sampler=sampler, batch_size=self.batch_size,
                                           num_workers=self.workers, pin_memory=True, drop_last=True)

    def get_test_dataset(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.workers, pin_memory=True, drop_last=True)

    def get_regular_dataset(self):
        return torch.utils.data.DataLoader(
            self.train_labeled_ds, batch_sampler=self.positive_indices.reshape(-1, 4),
            num_workers=self.workers, pin_memory=True)

    def get_positive_indices(self):
        return self.positive_indices
