# python evaluate.py -is 96 -bs 288 -os 12
import os
import sys
import utils
import torch
import warnings
import argparse

import numpy as np
import pandas as pd
import datetime as dt

from tqdm import tqdm
from datetime import timedelta
from model.seqseq import Model
from utils import gen_covariates
from matplotlib import pyplot as plt

sys.path.append('.')

warnings.filterwarnings("ignore")


def flatten(l):
    return [item for sublist in l for item in sublist]


def test(model, Dte):
    print(model)
    pre, true = [], []
    rmse, mae, mape, r2 = 0, 0, 0, 0
    model.eval()
    for i, (seq, target) in enumerate(tqdm(Dte)):
        seq_true = seq.to(device)
        seq_pred = model(seq_true)
        predict = seq_pred.cpu().detach().numpy().reshape(-1) * std + me  # 预测值去掉梯度并转换为numpy
        value = target.cpu().detach().numpy().reshape(-1) * std + me
        pre.append(predict)
        true.append(value)

    pre = flatten(pre)  # 展开列表

    true = flatten(true)
    rmse += utils.RMSE(np.array(pre), np.array(true))
    mae += utils.MAE(np.array(pre), np.array(true))
    mape += utils.SMAPE(np.array(pre), np.array(true))
    r2 += utils.R2(np.array(pre), np.array(true))

    folder = os.path.dirname(__file__)

    plot = os.path.join(folder, 'plot')
    if os.path.exists(plot):
        print('saving folder is already......')
    else:
        os.makedirs(plot)

    print('RMSE:', rmse, 'MAE:', mae, 'SMAPE:', mape, 'R2', r2)
    ch = np.random.randint(len(Dte))
    for i, (seq, target) in enumerate(tqdm(Dte)):
        if i == ch:
            f = plt.figure(figsize=(8, 20))
            nrows = 10
            ncols = 1
            ax = f.subplots(nrows, ncols)
            for k in range(nrows):
                x = np.random.randint(target.shape[0])
                seq_true = seq.to(device)
                seq_pred = model(seq_true)

                predict = seq_pred[x].cpu().detach().numpy() * std + me
                real = target[x].cpu().detach().numpy() * std + me
                predict = predict.reshape(-1, 1)
                real = real.reshape(-1, 1)
                ax[k].plot(predict, color='r', label='prediction')
                ax[k].plot(real, color='b', label='true')
            plot_path = plot + '/' + '1' + '.png'
            f.savefig(plot_path)
            plt.close()
    plt.figure(figsize=(12, 5))
    plt.plot(pre, 'r', label='prediction')
    plt.plot(true, 'b', label='true')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-fm", type=str, default='Caltech_kwh.csv')
    parser.add_argument("--save_name", "-sp", default="weight")
    parser.add_argument("--input_size", "-is", type=int, default=96)
    parser.add_argument("--output_size", "-os", type=int, default=12)
    parser.add_argument("--batch_size", "-bs", type=int, default=288)
    parser.add_argument("--hidden_size", "-hs", type=int, default=60)
    parser.add_argument("--long_interval", "-long_interval", type=int, default=15)
    parser.add_argument("--short_interval", "-short_interval", type=int, default=1)
    parser.add_argument("--Attention_hidden_size", "-ahs", type=int, default=10)
    parser.add_argument("--num_layers", "-nl", type=int, default=2)
    parser.add_argument("--cov", "-cov", type=int, default=4)
    parser.add_argument("--decoder_input_feature", "-df", type=int, default=1)
    parser.add_argument("--CNN_hid1", "-cn", type=int, default=5)
    parser.add_argument("--kernel1", "-kernel", type=int, default=3)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = os.path.dirname(__file__)
    model_dir = os.path.join(folder, args.save_name)
    filename = args.name
    data_path = utils.get_data_path()

    sample_period = args.long_interval
    test_start = dt.datetime(2020, 1, 1, 0, 0)
    test_end = dt.datetime(2020, 3, 31, 23, 60 - sample_period)
    data = pd.read_csv(os.path.join(data_path, filename), index_col='time')

    data.index = pd.to_datetime(data.index)

    ev_load = data['kwh']

    choose = ev_load[test_start:test_end - timedelta(minutes=1)]  # 从训练集开始到测试集结束的原始数据

    transform_series = []
    long_interval = (str(args.long_interval) + 'T')
    short_interval = (str(args.short_interval) + 'T')
    long_interval_series = choose.resample(long_interval, label='right').sum()  # 稀疏序列
    short_interval_series = choose.resample(short_interval, label='left').sum()  # 紧密序列
    k = args.long_interval // args.short_interval
    for i in range(0, short_interval_series.shape[0], k):  # transform
        transform_series.append(short_interval_series.iloc[i:i + k].values)
    transform_series = np.array(transform_series).reshape(-1, k)

    covariates = gen_covariates(long_interval_series.index)
    covariates = np.concatenate([covariates, transform_series], axis=1)
    covariates = pd.DataFrame(covariates)
    covariates.index = long_interval_series.index
    new = pd.concat([covariates, long_interval_series], axis=1)  # 0min对应10min的的协变量与0，1.。。。9
    new = new.fillna(0)

    me = new.iloc[:, -1].mean()
    std = new.iloc[:, -1].std()  # 标准化
    new.iloc[:, -1] = (new.iloc[:, -1] - me) / std

    test_set = new[test_start:test_end].values
    print('len of test:', test_set.shape)

    Dte = utils.process(test_set, args.batch_size, args.input_size, args.output_size)

    model = Model(args).to(device)
    utils.load_checkpoint(os.path.join(model_dir, f"input_w_{str(args.input_size)}_out_w_{str(args.output_size)}.pth.tar"), model)
    test(model, Dte)