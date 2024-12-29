import os
import json
import torch
import shutil
import logging
import numpy as np

from tqdm import tqdm
from scipy import stats
from sklearn import metrics

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def process(data, batch_size, input_size, output_size):
    load = data[:, -1]
    load = load.tolist()
    seq = []
    w_size = input_size + output_size
    for i in range(0, data.shape[0] - w_size, output_size):
        train_seq = []
        train_label = []
        for j in range(i, i + input_size):
            train_seq.append(data[j, :])
        train_label.append(load[i + input_size:i + w_size])
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))

    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0)
    return seq


def gen_covariates(times):
    covariates = np.zeros((times.shape[0], 4))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.weekday()
        covariates[i, 1] = input_time.hour
        covariates[i, 2] = input_time.month
        covariates[i, 3] = input_time.hour * 60 + input_time.minute
    for i in range(0, 4):
        covariates[:, i] = stats.zscore(covariates[:, i])
    return covariates


def get_data_path():
    folder = os.path.dirname(__file__)
    return os.path.join(folder, "data")


def RMSE(ypred, ytrue):
    ypred = ypred.reshape(-1)
    ytrue = ytrue.reshape(-1)
    # print(ypred.shape)
    rse = np.sum(((ypred-ytrue)**2/len(ypred)))**0.5
    return rse


def MAE(ypred,ytrue): #sum(|pred - true|)/n
    mae = metrics.mean_absolute_error(ypred, ytrue)
    return mae


def SMAPE(ypred,ytrue):#计算smape
    ypred = ypred.reshape(-1)
    ytrue = ytrue.reshape(-1)
    mape = np.sum(np.abs(ypred - ytrue) / (np.abs(ytrue) + np.abs(ypred) / 2) ) / len(ypred)
    return mape

def R2(ypred,ytrue):
    sse = np.sum((ytrue - ypred) ** 2)
    sst = np.sum((ytrue - np.mean(ypred)) ** 2)
    # print('sse', sse, 'sst', sst, (sse / sst), (1 - (sse/sst)))
    r2 = 1 - (sse / sst)  # r2_score(y_actual, y_predicted, multioutput='raw_values')
    return r2


def flatten(l):
    return [item for sublist in l for item in sublist]


# computing hammingLoss
def HammingLossClass(preLabels,test_targets):
    num_class,num_instance = np.mat(test_targets).shape
    temp = sum((preLabels != test_targets))
    miss_pairs = sum(temp)
    hammingLoss = miss_pairs/(num_class*num_instance)
    return hammingLoss


class Params:
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        '''Loads parameters from json file'''
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__


class RunningAverage:
    '''A simple class that maintains the running average of a quantity
    Example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    '''

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, epoch, checkpoint, save_epoch=False, save_name=""):
    '''
    Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    if save_epoch:
        torch.save(state, filepath)
    if is_best:
        torch.save(state, os.path.join(checkpoint, save_name + '.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None, save_name=""):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        # print('1*'*10)
        optimizer.load_state_dict(checkpoint['optim_dict'])
    # print(checkpoint)
    # print(type(optimizer.load_state_dict(checkpoint['optim_dict'])))
    # print(checkpoint['state_dict'])
    # print(checkpoint['optim_dict'])
    return checkpoint


def plot_all_epoch(variable, save_name, location='./figures/'):
    # print('variable shape:',variable.shape)
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()


def init_metrics(sample=True):
    metrics = {
        'MAE': np.zeros(2),  # numerator, denominator
        'RMSE': np.zeros(2),  # numerator, denominator, time step count
        'SMAPE':np.zeros(2),
        'R2':np.zeros(2),
        'test_loss': np.zeros(2),
    }
    if sample:
        metrics['rou90'] = np.zeros(2)
        metrics['rou50'] = np.zeros(2)
    return metrics


def final_metrics(raw_metrics, sampling=False):
    summary_metric = {}
    summary_metric['MAE'] = raw_metrics['MAE'][0] / raw_metrics['MAE'][1]
    summary_metric['SMAPE'] = raw_metrics['SMAPE'][0] / raw_metrics['SMAPE'][1]
    summary_metric['RMSE'] = raw_metrics['RMSE'][0] / raw_metrics['RMSE'][1]
    summary_metric['R2'] = raw_metrics['R2'][0] / raw_metrics['R2'][1]
    summary_metric['test_loss'] = (raw_metrics['test_loss'][0] / raw_metrics['test_loss'][1]).item()
    if sampling:
        summary_metric['rou90'] = raw_metrics['rou90'][0] / raw_metrics['rou90'][1]
        summary_metric['rou50'] = raw_metrics['rou50'][0] / raw_metrics['rou50'][1]
    return summary_metric
