import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.GYMNet_model import GYMData, GYMnet
from src.utils import load_data, select_meta

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get the arguments for the training process
argparser = argparse.ArgumentParser()

argparser.add_argument('--meta_path', type=str)
argparser.add_argument('--feats_path', type=str)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--batch_size', type=int, default=16)
argparser.add_argument('--nfeat_1', type=int, default=384)
argparser.add_argument('--nfeat_2', type=int, default=256)
argparser.add_argument('--drop', type=float, default=0.1)
argparser.add_argument('--l', type=float, default=2)

args = argparser.parse_args()

def get_data(batch_size): 
    # download data for all possible MoAs
    meta_train, feats_train = load_data(args.meta_path, args.feats_path)

    # select only the MoAs of interest
    # and encode them as numbers
    moa = ['Lipids', 'Respiration', 'Cytoskeleton', 'GPI', 'DMSO']

    moa_dict = {moa[i] : int(i) for i in range(len(moa))}

    meta_train, feats_train = select_meta(meta_train, feats_train, moa, 'MoA') 

    meta_train['moa_coded'] = meta_train.apply(lambda x: moa_dict[x['MoA']], axis=1)

    # define the dataset and make train/valid split
    ds = GYMData(feats_train, meta_train, nmoa=len(moa), l=args.l)

    ntrain = int(0.8 * len(ds))

    train_set, val_set = torch.utils.data.random_split(ds, [ntrain, len(ds) - ntrain])

    # # calculate the weights for the loss function
    # # to account for the class imbalance
    moa_weights = torch.zeros(len(moa))

    for i, ds_ in tqdm(enumerate(train_set)): 
        moa_weights += ds_['moa']

    moa_weights = 1 - moa_weights/len(train_set)

    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=64, shuffle=True)

    return train_dl, val_dl, moa_weights

# train for one epoch
def train_one_epoch(net, train_dl, optimizer, criterion, epoch_index, tb_writer):
    net.to(device)
    running_loss = 0.0
    last_loss = 0.0

    for i, data in tqdm(enumerate(train_dl)):
        inputs, adj, labels = data['feats'].to(device), data['adj_matrix'].to(device), data['moa'].to(device)

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(inputs, adj)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        running_loss += loss.item()

    last_loss = running_loss / len(train_dl) # loss per batch
    print('  epoch {} loss: {}'.format(i + 1, last_loss))
    tb_x = epoch_index * len(train_dl) + i + 1
    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    running_loss = 0.

    return last_loss


if __name__ == "__main__":

    # fetch the data
    train_dl, val_dl, moa_weights = get_data(args.batch_size)

    # define the network
    net = GYMnet(args.nfeat_1, args.nfeat_2, len(moa_weights), dropout=args.drop)

    # define optimizer, etc
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=moa_weights.to(device))

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    path = 'runs/gcn_site_average_{}'.format(timestamp)
    writer = SummaryWriter(path)
    json.dump(args.__dict__, open(path + '/args.json', 'w')) # save the parameters of the training

    epoch_number = 0

    EPOCHS = args.epochs 

    best_vloss = 1_000_000.
    best_vacc = 0.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        net.train(True)
        avg_loss = train_one_epoch(net, train_dl, optimizer, criterion, epoch_number, writer)

        running_vloss = 0.0
        total = 0.0
        correct = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        net.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dl):
                vinputs, vadj, vlabels = vdata['feats'].to(device), vdata['adj_matrix'].to(device), vdata['moa'].to(device)
                voutputs = net(vinputs, vadj)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

                # calcualting accuracy
                predicted = torch.argmax(voutputs, dim=1)
                total += vlabels.size(0)

                correct += (predicted == vlabels).sum().item()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACCURACY {}'.format(correct / total))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss, 
                         'Accuracy' : correct / total },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if correct / total > best_vacc:
        # if avg_vloss < best_vloss:
            best_vacc = correct / total
            best_vloss = avg_vloss
            model_path = os.path.join(path, 'model_{}_{}_{:.2f}'.format(timestamp, epoch_number+1, best_vacc))
            torch.save(net.state_dict(), model_path)

        epoch_number += 1

    torch.save(net.state_dict(), os.path.join(path, "model.pth"))



