import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
import math
import sys
import os
import shutil
from torch_ACA import odesolve#_endtime as odesolve

def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 60:
        optim_factor = 2
    elif epoch > 30:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

# from torch_ode_cifar import odesolve

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'sqnxt'], default='resnet')
parser.add_argument('--method', type=str, choices=['Euler', 'RK2', 'RK4', 'RK12','RK23', 'RK45'],
                    default='RK45')
parser.add_argument('--num_epochs', type=int, default=350)
parser.add_argument('--start_epoch', type=int, default=0)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='./checkpoint_multieval_RK12_resnet/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--h', type=float, default=0.2, help='Initial Stepsize')
parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
parser.add_argument('--t1', type=float, default=1.0, help='End time')
parser.add_argument('--rtol', type=float, default=1e-2, help='Releative tolerance')
parser.add_argument('--atol', type=float, default=1e-2, help='Absolute tolerance')
parser.add_argument('--print_neval', type=bool, default=False, help='Print number of evaluation or not')
parser.add_argument('--neval_max', type=int, default=50000, help='Maximum number of evaluation in integration')

parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()
if args.network == 'sqnxt':
    from models.sqnxt import SqNxt_23_1x

elif args.network == 'resnet':
    from models.resnet import ResNet18

num_epochs = int(args.num_epochs)
lr = float(args.lr)
start_epoch = int(args.start_epoch)
batch_size = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        self.options.update({'method': args.method})
        self.options.update({'h': args.h})
        self.options.update({'t0': args.t0})
        self.options.update({'t1': args.t1})
        self.options.update({'rtol': args.rtol})
        self.options.update({'atol': args.atol})
        self.options.update({'print_neval': args.print_neval})
        self.options.update({'neval_max': args.neval_max})
        #self.options.update({'t_eval': [args.t1]})
        print(self.options)

    def forward(self, x):
        out = odesolve(self.odefunc, x, self.options)
        return out[0]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', transform=transform_train, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', transform=transform_test, train=False, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=False)

if args.network == 'sqnxt':
    net = SqNxt_23_1x(10, ODEBlock)
elif args.network == 'resnet':
    net = ResNet18(ODEBlock)

net.apply(conv_init)
print(net)

print('Total number of parameters: {}'.format(sum(p.numel() for p in net.parameters())))

if is_use_cuda:
    net.to(device)
    net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()

        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                         % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size,
                            test_loss / (100 * (idx + 1)), correct / total))
        sys.stdout.flush()

    acc = correct / total
    return acc


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


best_acc = 0.0

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for _epoch in range(1):
    start_time = time.time()

    _lr = lr_schedule(args.lr, _epoch)
    adjust_learning_rate(optimizer, _lr)

    test_acc = test(_epoch)
    print()
    print()
    end_time = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

    # save model


print('Best Acc@1: %.4f\n' % (best_acc * 100))
print('Test Acc@1: %.4f\n' % (test_acc * 100))
