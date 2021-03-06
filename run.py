from __future__ import print_function
import argparse
import pickle

import torch
import torch.optim as optim
from gcommand_loader_wav import GCommandLoader
import numpy as np
from model import LeNet, VGG
from train import train, test
from attacks import attack,attack_no_loader
import os

# Training settings
parser = argparse.ArgumentParser(
    description='ConvNets for Speech Commands Recognition')
parser.add_argument('--train_path', default='gcommands/train',
                    help='path to the train data folder')
parser.add_argument('--test_path', default='gcommands/test',
                    help='path to the test data folder')
parser.add_argument('--valid_path', default='gcommands/valid',
                    help='path to the valid data folder')
parser.add_argument('--batch_size', type=int, default=100,
                    metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100,
                    metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='LeNet',
                    help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--epochs', type=int, default=100,
                    metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam',
                    help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234,
                    metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='num of batches to wait until logging train status')
parser.add_argument('--patience', type=int, default=5, metavar='N',
                    help='how many epochs of no loss improvement should we wait before stop training')

# feature extraction options
parser.add_argument('--window_size', default=.02,
                    help='window size for the stft')
parser.add_argument('--window_stride', default=.01,
                    help='window stride for the stft')
parser.add_argument('--window_type', default='hamming',
                    help='window type for the stft')
parser.add_argument('--normalize', default=True,
                    help='boolean, wheather or not to normalize the spect')


parser.add_argument('--test_mode', default=False, help='Whether to run model for test only or not')
parser.add_argument('--chkpt_path', default="", help='checkpoint path to load')


args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# loading data
train_dataset = GCommandLoader(args.train_path, window_size=args.window_size, window_stride=args.window_stride,
                               window_type=args.window_type, normalize=args.normalize)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=20, pin_memory=args.cuda, sampler=None)

valid_dataset = GCommandLoader(args.valid_path, window_size=args.window_size, window_stride=args.window_stride,
                               window_type=args.window_type, normalize=args.normalize)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

test_dataset = GCommandLoader(args.test_path, window_size=args.window_size, window_stride=args.window_stride,
                              window_type=args.window_type, normalize=args.normalize)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

# build model
if args.arc == 'LeNet':
    model = LeNet()
elif args.arc.startswith('VGG'):
    model = VGG(args.arc)
else:
    model = LeNet()

if args.cuda:
    model = model.cuda()

# define optimizer
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

best_valid_loss = np.inf
iteration = 0
epoch = 1

if args.test_mode:
    checkpoint = torch.load(args.chkpt_path)
    model.load_state_dict(checkpoint['net'].state_dict())
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

# attack model
attack(test_loader, 0, model)

# trainint with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience) and not args.test_mode:
    train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval)
    # attack model
    attack(test_loader, 0, model)
    valid_loss = test(valid_loader, model, args.cuda)
    if valid_loss > best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        print('Saving model...')
        iteration = 0
        best_valid_loss = valid_loss
        state = {
            'net': model,
            'acc': valid_loss,
            'epoch': epoch,
            'optimizer_state_dict:': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt2.t7')
    epoch += 1

attack(test_loader, 0, model)
# attack model
# with open("idxs", "rb") as f:
#     idxs = pickle.load(f)

# for i in idxs:
#     try:
#         attack(test_loader, i, model, save=True)#4250
#     except:
#         continue

# attack_no_loader(test_loader, model)
# test model
test(test_loader, model, args.cuda)
