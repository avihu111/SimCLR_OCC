import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import neptune.new as neptune
from config import params
neptune_run = neptune.init(project='avihu/simCLR',
                           api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyODYzMGNiMC0yNzk3LTQ0MmYtYWVkMy0wNDgyMTIzMzk5NjkifQ==',
                           source_files=['*.py', 'requirements.txt'])
neptune_run['parameters'] = params
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
# the class I wish to make it's representations compact


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        print("shoot, only cpu :(")
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset('./datasets')

    train_dataset = dataset.get_dataset(params['dataset_name'], args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    last_valid_index = (train_dataset.dataset.labels == params['relevant_class']).nonzero()[0][params['num_examples']] + 1
    model = ResNetSimCLR(base_model=params['arch'], out_dim=params['out_dim'])
    pretrained = torch.load('./models/checkpoint_0100.pth.tar')
    print('loading pretrained {} with {} epochs'.format(pretrained['arch'], pretrained['epoch']))
    model.load_state_dict(pretrained['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'], weight_decay=params['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    test_dataset = dataset.get_test_dataset(params['dataset_name'])

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=1, pin_memory=True, drop_last=True)

    positive_indices = (train_dataset.dataset.labels == 0).nonzero()[0][:params['num_examples']]

    train_labeled_dataset = dataset.get_test_dataset('stl10_train')
    train_labeled_loader = torch.utils.data.DataLoader(
        train_labeled_dataset, batch_sampler=positive_indices.reshape(2, -1),
        num_workers=1, pin_memory=True)
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args,
                        last_valid_index=last_valid_index, neptune_run=neptune_run)
        simclr.train(train_loader, test_loader, train_labeled_loader)


if __name__ == "__main__":
    main()
