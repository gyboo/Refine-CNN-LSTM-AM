# python train.py -e 1 -is 96 -bs 288 -os 12

import os
import sys
import utils
import torch
import warnings
import argparse

import numpy as np
import pandas as pd
import datetime as dt

from torch import nn
from tqdm import tqdm
from datetime import timedelta
from model.seqseq import Model
from utils import gen_covariates
from torch.optim.lr_scheduler import StepLR
sys.path.append('.')

warnings.filterwarnings("ignore")


def train(args, Dtr, Val, save_dir):

    loss_function = nn.MSELoss().to(device)

    model = Model(args).to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=0.9, )
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    best_loss = 1e6
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)

            label = label.to(device)
            pred = model(seq)
            loss = loss_function(pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # validation
        val_losses = []
        model = model.to(device)
        with torch.no_grad():
            for i, (valseq, valabel) in enumerate(Val):

                seq_true = valseq.to(device)
                seq_pred = model(seq_true)

                valabel = valabel.to(device)
                loss = loss_function(seq_pred, valabel).to(device)
                val_losses.append(loss.item())

        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_losses)
        is_best = val_loss <= best_loss
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              save_epoch=False,
                              epoch=epoch,
                              is_best=is_best,
                              save_name=f"input_w_{str(args.input_size)}_out_w_{str(args.output_size)}",
                              checkpoint=save_dir)
        if(epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}: train loss {train_loss} val loss {val_loss} lr {scheduler.get_lr()[0]}')


def flatten(l):
    return [item for sublist in l for item in sublist]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-fm", type=str, default='Caltech_kwh.csv')
    parser.add_argument("--save_name", "-sp", default="weight")
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--input_size", "-is", type=int, default=96)
    parser.add_argument("--output_size", "-os", type=int, default=12)
    parser.add_argument("--batch_size", "-bs", type=int, default=288)
    parser.add_argument("--long_interval", "-long_interval", type=int, default=15)
    parser.add_argument("--short_interval", "-short_interval", type=int, default=1)
    parser.add_argument("--hidden_size", "-hs", type=int, default=60)
    parser.add_argument("--Attention_hidden_size", "-ahs", type=int, default=10)
    parser.add_argument("--num_layers", "-nl", type=int, default=2)
    parser.add_argument("--optimizer", "-op", type=str, default='adam')
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    parser.add_argument("--cov", "-cov", type=int, default=4)
    parser.add_argument("--decoder_input_feature", "-df", type=int, default=1)
    parser.add_argument("--step_size", "-ss", type=int, default=30)
    parser.add_argument("--CNN_hid1", "-cn", type=int, default=5)
    parser.add_argument("--kernel1", "-kernel", type=int, default=3)
    parser.add_argument("--gamma", "-gm", type=float, default=0.9)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = os.path.dirname(__file__)
    model_dir = os.path.join(folder, args.save_name)

    filename = args.name
    data_path = utils.get_data_path()

    sample_period = args.long_interval
    train_start = dt.datetime(2018, 4, 25, 0, 0)  # 设定训练集的开始时间
    train_end = dt.datetime(2019, 8, 31, 23, 60 - sample_period)
    val_start = dt.datetime(2019, 9, 1, 0, 0)
    val_end = dt.datetime(2019, 12, 31, 23, 60 - sample_period)
    data = pd.read_csv(os.path.join(data_path, filename), index_col='time')
    data.index = pd.to_datetime(data.index)
    ev_load = data['kwh']
    choose = ev_load[train_start:val_end - timedelta(minutes=1)]  # 从训练集开始到验证集结束的原始数据

    transform_series = []
    long_interval = (str(args.long_interval) + 'T')
    short_interval = (str(args.short_interval) + 'T')

    long_interval_series = choose.resample(long_interval, label='right').sum()  # long interval time series
    short_interval_series = choose.resample(short_interval, label='left').sum()  # short interval time series
    k = args.long_interval // args.short_interval
    for i in range(0, short_interval_series.shape[0], k):   # transform
        transform_series.append(short_interval_series.iloc[i:i + k].values)
    transform_series = np.array(transform_series).reshape(-1, k)

    covariates = gen_covariates(long_interval_series.index)  # add covariates: hour,minute... of timestamp
    covariates = np.concatenate([covariates, transform_series], axis=1)
    covariates = pd.DataFrame(covariates)
    covariates.index = long_interval_series.index
    new = pd.concat([covariates, long_interval_series], axis=1)
    new = new.fillna(0)

    me = new.iloc[:, -1].mean()
    std = new.iloc[:, -1].std()  # 标准化
    new.iloc[:, -1] = (new.iloc[:, -1] - me) / std
    train_set = new[train_start:train_end].values
    val_set = new[val_start:val_end].values

    print('len of train', train_set.shape, 'len of val:', val_set.shape)
    Dtr = utils.process(train_set, args.batch_size, args.input_size, args.output_size)
    Val = utils.process(val_set, args.batch_size, args.input_size, args.output_size)

    train(args, Dtr, Val, model_dir)
