import argparse
import time
import os
import numpy as np
import pandas as pd
import torch

import math
from tqdm import tqdm
import h5py
import scipy.sparse as sp

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import dgl
from ogb.nodeproppred import Evaluator

import sys
from models import GAT
from utils1 import *

import warnings

warnings.filterwarnings('ignore')

version = None
epsilon = 1 - math.log(2)
device = None

n_node_feats, n_classes = 0, 0


def load_data(args, dataset, filecode):
    global n_node_feats, n_classes

    filename = 'sample_result_({}).mat'.format(filecode)

    path = './'
    print('Loading {} {} dataset...'.format(dataset[:4], filename[:-4]), 'r+')
    dict_data = h5py.File('{}/mat/{}{}'.format(path, dataset, filename))
    features = dict_data['scores']
    adj = dict_data['subnetwork_adjacency']
    labels = pd.read_table('{}/label/{}{}.txt'.format(path, dataset, filecode), header=None)

    features = sp.csr_matrix(features, dtype=np.float32).T
    adj = sp.coo_matrix(adj, dtype=np.float32)

    print("done")

    data = dgl.from_scipy(adj)
    features = torch.FloatTensor(features.todense())

    data.ndata['feat'] = features
    # --------------------------------

    # preprocess
    evaluator = Evaluator(name=dataset)
    graph = data

    # the original type of labels array is string
    labels = np.array(labels, dtype=np.int32)

    posi_example = np.argwhere(labels == 1)
    posi_example[:, 1] = 1
    neg_example = np.argwhere(labels == 0)
    tr_example = np.argwhere(labels == -1)
    labels[np.where(labels == -1)] = 0

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = 2
    pred_mx = torch.zeros(labels.shape)

    return graph, labels, posi_example, neg_example, tr_example, evaluator, pred_mx


def preprocess(graph):
    global n_node_feats

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print("Total edges before adding self-loop {}".format(graph.number_of_edges()))
    graph = graph.remove_self_loop().add_self_loop()
    print("Total edges after adding self-loop {}".format(graph.number_of_edges()))

    return graph


def gen_model(args):
    model = GAT(
        n_node_feats + n_classes if args.use_labels else n_node_feats,
        n_classes,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        use_symmetric_norm=args.use_norm,
    )

    return model


def custom_loss_function(x, labels, weight):
    loss_weight = torch.tensor([weight, 1.0], dtype=torch.float32)
    loss_weight = loss_weight.cuda()
    if len(x.shape) > 2:
        x = x.reshape((-1, 2))

    labels = labels.reshape((-1))
    criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=True)
    criterion.cuda()

    y = criterion(x, labels)
    return torch.mean(y)


def train(model, graph, labels, train_idx, val_idx, test_idx, optimizer, use_labels, n_label_iters, args):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = torch.rand(train_idx.shape) < mask_rate
        train_labels_idx = train_idx[mask]
        train_test_idx = train_idx[~mask]
        feat = add_labels(n_classes, device, feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = torch.rand(train_idx.shape) < mask_rate

        train_test_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)

    if n_label_iters > 0:
        test_idx = torch.reshape(test_idx, (-1,))
        unlabel_idx = torch.cat([train_test_idx, val_idx[:, 0], test_idx])
        for _ in range(n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx].detach(), dim=-1)
            pred = model(graph, feat)

    ce_loss = custom_loss_function(pred[train_test_idx], labels[train_test_idx], args.weight)
    loss = ce_loss

    loss.backward()
    optimizer.step()

    return pred.detach()


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running, pred_mx):
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0

    for epoch in tqdm(range(1, args.n_epochs + 1), desc="running {}".format(n_running), ncols=80):
        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        pred = train(
            model, graph, labels, train_idx, val_idx, test_idx, optimizer, args.use_labels,
            args.n_label_iters, args,
        )
        acc = compute_acc(pred, labels[test_idx], test_idx, pred_mx, True)

        toc = time.time()
        total_time += toc - tic

    return pred_mx


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon
    # args
    argparser = argparse.ArgumentParser("GAT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, help="running times", default=5)
    argparser.add_argument("--n-epochs", type=int, help="number of epochs", default=2000)

    argparser.add_argument("--use-labels", type=str2bool, default=True)
    argparser.add_argument("--n-label-iters", type=int, help="number of label iterations", default=1)
    argparser.add_argument("--no-attn-dst", type=str2bool, default=True)
    argparser.add_argument("--use-norm", type=str2bool, default=True)
    argparser.add_argument('--topo-loss-ratio', type=float, default=0.8)
    argparser.add_argument('--topo-mask-threshold', type=float, default=0.2)

    argparser.add_argument("--n-layers", type=int, help="number of layers", default=3)
    argparser.add_argument("--n-heads", type=int, help="number of heads", default=5)
    argparser.add_argument("--n-hidden", type=int, help="number of hidden units", default=750)

    argparser.add_argument("--dropout", type=float, help="dropout rate", default=0.75)
    argparser.add_argument("--input-drop", type=float, help="input drop rate", default=0.0)
    argparser.add_argument("--attn-drop", type=float, help="attention dropout rate", default=0.25)
    argparser.add_argument("--edge-drop", type=float, help="edge drop rate", default=0.25)

    argparser.add_argument("--lr", type=float, help="learning rate", default=0.02)
    argparser.add_argument("--wd", type=float, help="weight decay", default=1e-6)
    argparser.add_argument("--log-every", type=int, help="log every LOG_EVERY epochs", default=100)

    argparser.add_argument("--dataset", type=str, default="BRCA/")
    argparser.add_argument("--filecode", type=int, default=1)
    argparser.add_argument("--weight", type=float, default=3.0)

    argparser.add_argument("--version", type=str, default="PersonalizedGNN")
    args = argparser.parse_args()

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    # config GPU
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load data
    graph, labels, posi_example, neg_example, tr_example, evaluator, pred_mx = load_data(
        args, args.dataset, args.filecode)
    graph = preprocess(graph)

    graph = graph.to(device)
    labels = torch.LongTensor(labels).to(device)

    # run
    posi = posi_example.shape[0] // 5
    neg = neg_example.shape[0] // 5
    tr = tr_example.shape[0] // 5
    args.weight = neg / posi
    # divide dataset
    for n_running in range(1, args.n_runs + 1):
        if n_running == 0:
            train_idx = np.concatenate((posi_example[:posi * 3], neg_example[:neg * 3]))
            val_idx = np.concatenate((posi_example[posi * 3:posi * 4], neg_example[neg * 3:neg * 4]))
            test_idx = np.concatenate((posi_example[posi * 4:posi * 5], neg_example[neg * 4:neg * 5], tr_example))
        elif n_running == 1:
            train_idx = np.concatenate((posi_example[posi:posi * 4], neg_example[neg:neg * 4]))
            val_idx = np.concatenate((posi_example[posi * 4:posi * 5], neg_example[neg * 4:neg * 5]))
            test_idx = np.concatenate((posi_example[:posi], neg_example[:neg], tr_example[tr:tr * 2]))
        elif n_running == 2:
            train_idx = np.concatenate((posi_example[posi * 2:posi * 5], neg_example[neg * 2:neg * 5]))
            val_idx = np.concatenate((posi_example[:posi], neg_example[:neg]))
            test_idx = np.concatenate((posi_example[posi:posi * 2], neg_example[neg:neg * 2], tr_example))
        elif n_running == 3:
            train_idx = np.concatenate(
                (posi_example[:posi], posi_example[posi * 3:posi * 5], neg_example[:neg], neg_example[neg * 3:neg * 5]))
            val_idx = np.concatenate((posi_example[posi:posi * 2], neg_example[neg:neg * 2]))
            test_idx = np.concatenate((posi_example[posi * 2:posi * 3], neg_example[neg * 2:neg * 3], tr_example))
        else:
            train_idx = np.concatenate((posi_example[:posi * 2], posi_example[posi * 4:posi * 5], neg_example[:neg * 2],
                                        neg_example[neg * 4:neg * 5]))
            val_idx = np.concatenate((posi_example[posi * 2:posi * 3], neg_example[neg * 2:neg * 3]))
            test_idx = np.concatenate((posi_example[posi * 3:posi * 4], neg_example[neg * 3:neg * 4], tr_example))

        train_idx = torch.from_numpy(train_idx).to(torch.long)
        train_idx = torch.LongTensor(train_idx).to(device)
        val_idx = torch.from_numpy(val_idx).to(torch.long)
        val_idx = torch.LongTensor(val_idx).to(device)
        test_idx = torch.from_numpy(test_idx).to(torch.long)
        test_idx = torch.LongTensor(test_idx).to(device)

        seed = n_running - 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

        pred_mx = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running,
                      pred_mx)

    # ouotput raw score
    pred_mx = pred_mx.detach().cpu().numpy()
    pred_mx = np.array(pred_mx)
    pd.DataFrame(pred_mx).to_csv("./{}_{}.csv".format(args.dataset[:4], args.filecode))


if __name__ == "__main__":
    print(' '.join(sys.argv))
    main()
