import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import neptune.new as neptune

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
STL_CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
MOTORED = [0, 2, 8, 9]
ANIMALS = [1, 3, 4, 5, 6, 7]


parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10', 'caltech256'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
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
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
# the class I wish to make it's representations compact
parser.add_argument('--relevant-classes', default='american_flag', help='classes that are inliers ',
                    choices=['animals', 'motored', 'random_set', 'random_one', 'american_flag'])
parser.add_argument('--num-labeled-examples', default=40, type=int, help='number of labeled examples')
parser.add_argument('--lambda', default=10., type=float, help='lambda')
parser.add_argument('--debug', action='store_true', help='is debug mode')
parser.add_argument('--large_features', action='store_true', help='before last layer or last')


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

    params = vars(args)

    num_classes = 256 if 'caltech' in args.dataset_name else 10
    random_relevant_classes = np.arange(num_classes)[np.random.random(num_classes) < 0.1]
    while len(random_relevant_classes) == 0:
        random_relevant_classes = np.arange(num_classes)[np.random.random(num_classes) < 0.1]
    CLASSES_MAPPER = {'animals': ANIMALS, 'motored': MOTORED,
                      'random_set': random_relevant_classes,
                      'random_one': [np.random.choice(np.arange(num_classes))],
                      'american_flag': [1]}
    params['relevant_classes'] = CLASSES_MAPPER[args.relevant_classes]
    run_name = 'SimCLR_lambda={}'.format(params['lambda'])
    mode = 'debug' if args.debug else 'async'
    dataset = ContrastiveLearningDataset('./datasets', batch_size=params['batch_size'], workers=args.workers,
                                         relevant_classes=params['relevant_classes'],
                                         num_labeled_examples=params['num_labeled_examples'], name=params['dataset_name'])

    print(params)
    with torch.cuda.device(args.gpu_index):
        model = ResNetSimCLR(base_model=params['arch'], out_dim=params['out_dim'])
        folder = 'caltech256' if args.arch == 'resnet50' else 'stl10'
        pretrained_path = f'./models/{folder}/checkpoint_0100.pth.tar'
        try:
            pretrained = torch.load(pretrained_path, map_location=f'cuda:{args.gpu_index}')
            print('loading pretrained {} with {} epochs'.format(pretrained['arch'], pretrained['epoch']))
            model.load_state_dict(pretrained['state_dict'])
        except FileNotFoundError:
            pass
        train_loader = dataset.get_augmented_dataset()
        test_loader = dataset.get_test_dataset()
        train_labeled_loader = dataset.get_regular_dataset()
        positive_indices = dataset.get_positive_indices()
        optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
        # todo: complete these values
        neptune_run = neptune.init(name=run_name,
                                   project='***',
                                   api_token='***',
                                   source_files=['*.py', 'requirements.txt'], mode=mode)
        neptune_run['parameters'] = params
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args,
                        positive_indices=positive_indices, neptune_run=neptune_run, params=params)
        simclr.train(train_loader, test_loader, train_labeled_loader)


if __name__ == "__main__":
    main()
