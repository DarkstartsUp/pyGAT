from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from sklearn.metrics import mean_squared_error
from models import GAT, SpGAT, GAT_with_LSTM

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=50, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
data_root = '/media/luvision/新加卷1/out_graph_features/1-HIT_Canteen_frames'
data = np.load(os.path.join(data_root, 'train.npz'), allow_pickle=True)
features_train, adj_init_train, adj_gt_train = data['features'], data['init_adj'], data['gt_adj']  # .reshape((data['features'].shape[0], -1, 2))
data = np.load(os.path.join(data_root, 'val.npz'), allow_pickle=True)
features_val, adj_init_val, adj_gt_val = data['features'], data['init_adj'], data['gt_adj']
data = np.load(os.path.join(data_root, 'test.npz'), allow_pickle=True)
features_test, adj_init_test, adj_gt_test = data['features'], data['init_adj'], data['gt_adj']

# Model and optimizer
model = GAT_with_LSTM(nfeat=300,
            nhid=args.hidden,
            nclass=args.hidden,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
model.cuda()


def train(epoch):
    """
    Run at each epoch.
    """
    t = time.time()
    model.train()  # dropout is enabled under train mode
    optimizer.zero_grad()
    # output: [batch_size, log(softmax(x))]

    total_loss_train = 0
    total_acc_train = 0
    for it, feat_train in enumerate(features_train):
        a_ini_train = adj_init_train[it]
        a_gt_train = adj_gt_train[it]
        non_zero_ratio = len([a for a in a_gt_train.flatten().tolist() if a > 1e-5]) / len(a_gt_train.flatten())

        feat_train = Variable(torch.FloatTensor(feat_train).cuda())
        a_ini_train = Variable(torch.FloatTensor(a_ini_train).cuda())
        a_gt_train = Variable(torch.FloatTensor(a_gt_train).cuda())

        output, att = model(feat_train, a_ini_train, return_final_att=True)  # .reshape((feat_train.size()[0], -1))
        # calc loss
        mask = a_gt_train.ge(0.5)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not mask[i, j] and random.random() < non_zero_ratio:
                    mask[i, j] = True
        att_for_loss = torch.masked_select(att, mask)
        gt_for_loss = torch.masked_select(a_gt_train, mask)

        # loss_train = F.binary_cross_entropy(att, adj_gt_train)
        loss_train = F.binary_cross_entropy(att_for_loss, gt_for_loss)
        total_loss_train += loss_train

        # acc_train = mean_squared_error(att.cpu().flatten().detach().numpy(), adj_gt_train.cpu().flatten().detach().numpy())
        acc_train = mean_squared_error(att_for_loss.cpu().detach().numpy(), gt_for_loss.cpu().detach().numpy())
        total_acc_train += acc_train

        loss_train.backward()
        optimizer.step()

    # print('train zero: ', len([a for a in att.cpu().flatten().detach().numpy().tolist() if a > 1e-5]),
    #       len(att.cpu().flatten().detach().numpy()))

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        loss_val = 0
        acc_val = 0
        for it, feat_val in enumerate(features_val):
            a_ini_val = adj_init_val[it]
            a_gt_val = adj_gt_val[it]
            non_zero_ratio = len([a for a in a_gt_val.flatten().tolist() if a > 1e-5]) / len(a_gt_val.flatten())

            feat_val = Variable(torch.FloatTensor(feat_val).cuda())
            a_ini_val = Variable(torch.FloatTensor(a_ini_val).cuda())
            a_gt_val = Variable(torch.FloatTensor(a_gt_val).cuda())

            output, att = model(feat_val, a_ini_val, return_final_att=True)  # .reshape((feat_val.size()[0], -1))
            # calc loss
            mask = a_gt_val.ge(0.5)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if not mask[i, j] and random.random() < non_zero_ratio:
                        mask[i, j] = True
            att_for_loss = torch.masked_select(att, mask)
            gt_for_loss = torch.masked_select(a_gt_val, mask)

            loss_val += F.binary_cross_entropy(att_for_loss, gt_for_loss)
            acc_val += mean_squared_error(att_for_loss.cpu().detach().numpy(), gt_for_loss.cpu().detach().numpy())

    # loss_val = F.binary_cross_entropy(att, adj_gt_val)
    # acc_val = mean_squared_error(att.cpu().flatten().detach().numpy(), adj_gt_val.cpu().flatten().detach().numpy())
    # print('val zero: ', len([a for a in att.cpu().flatten().detach().numpy().tolist() if a > 1e-5]),
    #       len(att.cpu().flatten().detach().numpy()))
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(total_loss_train),
          'acc_train: {:.4f}'.format(total_acc_train),
          'loss_val: {:.4f}'.format(loss_val),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val


def compute_test():
    model.eval()
    loss_test = 0
    acc_test = 0
    for it, feat_test in enumerate(features_test):
        a_ini_test = adj_init_test[it]
        a_gt_test = adj_gt_test[it]
        non_zero_ratio = len([a for a in a_gt_test.flatten().tolist() if a > 1e-5]) / len(a_gt_test.flatten())

        feat_test = Variable(torch.FloatTensor(feat_test).cuda())
        a_ini_test = Variable(torch.FloatTensor(a_ini_test).cuda())
        a_gt_test = Variable(torch.FloatTensor(a_gt_test).cuda())

        output, att = model(feat_test, a_ini_test, return_final_att=True)   # .reshape((feat_test.size()[0], -1))
        # calc loss
        mask = a_gt_test.ge(0.5)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not mask[i, j] and random.random() < non_zero_ratio:
                    mask[i, j] = True
        att_for_loss = torch.masked_select(att, mask)
        gt_for_loss = torch.masked_select(a_gt_test, mask)

        loss_test += F.binary_cross_entropy(att_for_loss, gt_for_loss)
        acc_test += mean_squared_error(att_for_loss.cpu().detach().numpy(), gt_for_loss.cpu().detach().numpy())
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "accuracy= {:.4f}".format(acc_test))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), 'ckpts/{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('ckpts/*.pkl')
    for file in files:
        epoch_nb = int(file.split('/')[1].split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('ckpts/*.pkl')
for file in files:
    epoch_nb = int(file.split('/')[1].split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('ckpts/{}.pkl'.format(best_epoch)))

# Testing
compute_test()
