import numpy as np
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import Dataset
class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

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

    def get_dataset(self, name, n_views):
        if name == 'cifar10':
            ds = datasets.CIFAR10(self.root_folder, train=True,
                             transform=ContrastiveLearningViewGenerator(
                                 self.get_simclr_pipeline_transform(32),
                                 n_views),
                             download=True)
            ds.labels = np.array(ds.targets)
            return DatasetWrapper(ds)

        if name == 'stl10':
            ds = datasets.STL10(self.root_folder, split='train+unlabeled',
                           transform=ContrastiveLearningViewGenerator(
                               self.get_simclr_pipeline_transform(96),
                               n_views),
                           download=True)
            return DatasetWrapper(ds)
        raise NotImplemented(f'dataset {name} not supported')

    def get_test_dataset(self, name):
        if name == 'stl10':
            return datasets.STL10(self.root_folder, split='test',transform=transforms.ToTensor(), download=True)
        if name == 'stl10_train':
            return datasets.STL10(self.root_folder, split='train', transform=transforms.ToTensor(), download=True)
        if name == 'cifar10':
            ds = datasets.CIFAR10(self.root_folder, train=False, transform=transforms.ToTensor(), download=True)
            ds.labels = np.array(ds.targets)
            return ds
        if name == 'cifar10_train':
            ds = datasets.CIFAR10(self.root_folder, train=True, transform=transforms.ToTensor(), download=True)
            ds.labels = np.array(ds.targets)
            return ds

        raise ValueError("unsupported dataset")