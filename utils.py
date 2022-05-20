import torch
from sklearn.metrics import roc_auc_score
import os
import numpy as np
import pandas as pd
import logging
import time
import argparse


def backup_code(path, version):
    import os
    if not os.path.exists(path):
        os.mkdir(path)
    os.mkdir(f'{path}/{version}')
    os.system(F'rsync -avr --exclude-from=".gitignore" * {path}/{version}/')


def get_pred_confidence(pred, label, softmax_ed):
    with torch.no_grad():
        if not softmax_ed:
            pred = torch.nn.functional.softmax(pred, dim=1)
        confidence = pred[torch.arange(pred.shape[0]), label]
        return confidence.detach()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_logging_format(rank=None):
    if rank:
        logging.basicConfig(
            level=logging.INFO,
            format=F'rank{rank} %(asctime)s %(filename)s line%(lineno)d | %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(filename)s line%(lineno)d | %(levelname)s: %(message)s')


def formatted_time():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def add_labels(n_classes, device, feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes]).to(device)
    for i in idx:
        onehot[i, labels[i]] = 1
    return torch.cat([feat, onehot], dim=-1)


def compute_acc(pred, true, idx, pred_mx, importance=False):
    y_pred = np.squeeze(pred.argmax(dim=-1, keepdim=True)[idx].numpy()[:, 0])
    y_true = np.squeeze(true.numpy()[:, 0])
    result = 0.0
    idx = idx[:, 0]
    for i in range(y_pred.shape[0] - 1):
        if y_pred[i] == y_true[i]:
            result += 1.0
        if importance & y_pred[i] == 1:
            pred_mx[idx[i]] += 1

    result = result / y_pred.shape[0]
    return result


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def print_dict(d, num_tab=0, prefix=None):
    if num_tab == 0 and prefix is not None:
        print(prefix)
    for k, v in d.items():
        if not isinstance(v, dict):
            print('{}{}: {}'.format(num_tab * '\t' + '', k, v))
        else:
            print('{}{}:'.format(num_tab * '\t' + '', k))
            print_dict(v, num_tab + 1)


def write_dict(message_dict, file, num_tab=0, prefix=None):
    if num_tab == 0 and prefix is not None:
        file.write(prefix)
        file.write('\n')
    for k, v in message_dict.items():
        if not isinstance(v, dict):
            file.write('{}{}: {}\n'.format(num_tab * '\t' + '', k, v))
        else:
            file.write('{}{}:\n'.format(num_tab * '\t' + '', k))
            write_dict(v, file, num_tab + 1)


def hint_line(message):
    return '{} {} {}'.format('#' * 30, message, '#' * 30)
