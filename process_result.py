import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
from tqdm import trange
import warnings

warnings.filterwarnings('ignore')


def single_dataset(dataset):
    tot = {}
    # process [dataset] 1-11
    for filecode in trange(1, 11):
        path = './'
        filename = 'sample_result_({}).mat'.format(filecode)

        dict_data = h5py.File('{}{}{}'.format(path, dataset, filename))
        labels = pd.read_table('./label/{}{}.txt'.format(dataset, filecode), header=None)
        labels = np.squeeze(labels)
        grades = pd.read_csv('./{}_{}'.format(dataset[:4], filecode), header=1)
        grades = np.array(grades)[:, 1]

        subnet = dict_data['subnetwork_genes']
        gene_list = ''

        for i in range(subnet.shape[1]):
            gene = ''.join([chr(v[0]) for v in dict_data[(subnet[0][i])]])
            gene_list = gene_list + ',' + gene
        gene_list = np.array(gene_list.split(",")[1:])

        for i in range(grades.shape[0]):
            if (gene_list[i], labels[i]) not in tot:
                tot[(gene_list[i], labels[i])] = [grades[i]]
            else:
                tot[(gene_list[i], labels[i])].append(grades[i])

    mx = []
    # unlabelled genes are counted 5 times as much as labeled genes
    for key in tot.keys():
        if key[1] == 1:
            mx.append([key[0], key[1], np.mean(tot[key]) * 5])
        else:
            mx.append([key[0], key[1], np.mean(tot[key])])
    pd.DataFrame(mx).to_csv("./{}.csv".format(dataset[:-1]), header=0, index=False)


def top30(dataset):
    scores = []
    tot = {}
    # calculate the average precision of [dataset] 1-11
    for filecode in trange(1, 11):
        path = './'
        filename = 'sample_result_({}).mat'.format(filecode)
        dict_data = h5py.File('{}{}{}'.format(path, dataset, filename))
        labels = pd.read_table('./label/{}{}.txt'.format(dataset, filecode), header=None)
        labels = np.squeeze(labels)
        grades = pd.read_csv('./{}_{}.csv'.format(dataset[:4], filecode), header=1)
        grades = np.array(grades)[:, 1]

        subnet = dict_data['subnetwork_genes']
        gene_list = ''

        for i in range(subnet.shape[1]):
            gene = ''.join([chr(v[0]) for v in dict_data[(subnet[0][i])]])
            gene_list = gene_list + ',' + gene
        gene_list = np.array(gene_list.split(",")[1:])

        for i in range(grades.shape[0]):
            if (gene_list[i], labels[i]) not in tot:
                tot[(gene_list[i], labels[i])] = [grades[i]]
            else:
                tot[(gene_list[i], labels[i])].append(grades[i])

    mx = []
    for key in tot.keys():
        if key[1] == 1:
            mx.append([key[0], key[1], np.mean(tot[key]) * 5])
        else:
            mx.append([key[0], key[1], np.mean(tot[key])])
    mx = np.array(mx)[:, 1:]
    mx = np.array(mx, dtype=np.float32)
    mx = mx[np.lexsort(-mx.T)]
    mx = mx[:30, 0]
    mx[mx == -1] = 0
    score = np.zeros_like(mx)
    for i in range(30):
        score[i] = np.mean(mx[:i + 1])
    score.reshape((-1, 1))
    scores.append(score)
    pd.DataFrame(scores).to_csv("./{}.csv".format(dataset[:-1]), header=0, index=False)


if __name__ == "__main__":
    print("processing results")
    # single_dataset('BRCA/')
    # top30('BRCA/')
