import sys
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

def read_art_qrels(art_qrels):
    page_paras = {}
    with open(art_qrels, 'r') as f:
        for l in f:
            q = l.split(' ')[0]
            p = l.split(' ')[2]
            if q not in page_paras.keys():
                page_paras[q] = [p]
            else:
                page_paras[q].append(p)
    return page_paras

def prepare_batch(batch_queries, X_data, emb_size):
    X_batch = []
    true_batch_labels = []
    max_paralist_len = max([len(X_data[q]) - 1 for q in batch_queries])
    num_paras = []
    num_k = []
    mean_para_per_k = []

    for q in batch_queries:
        #print(q)
        true_batch_label = [d[2] for d in X_data[q][1:]] + [-1] * (max_paralist_len - len(X_data[q][1:]))
        num_clusters = len(list(set(true_batch_label)))
        num_paras.append(len(X_data[q][1:]))
        num_k.append(num_clusters)
        mean_para_per_k.append(len(X_data[q][1:])/num_clusters)
        true_batch_labels.append(true_batch_label)
        qvec = X_data[q][0]
        paraids = []
        paravecs = []
        paravec_labels = []
        for d in X_data[q][1:]:
            paraids.append(d[0])
            paravecs.append(d[1])
            paravec_labels.append(d[2])
        # print(q + " passage count: " + str(len(paraids)))
        for i in range(max_paralist_len - len(X_data[q][1:])):
            paraids.append("dummy"+str(i))
            paravecs.append(np.zeros(emb_size))
            #paravecs.append(np.full(emb_size, -99.0))
            paravec_labels.append(-1)
        X_cats = []
        parapair_ids = []
        num_cluster_vec = np.zeros(3*emb_size)
        num_cluster_vec[0] = num_clusters
        X_cats.append(num_cluster_vec)
        for i in range(len(paraids)):
            for j in range(len(paraids)):
                if i==j:
                    X_cats.append(np.hstack((qvec, paravecs[i], paravecs[i])))
                    parapair_ids.append(paraids[i] + '_' + paraids[i])
                else:
                    X_cats.append(np.hstack((qvec, paravecs[i], paravecs[j])))
                    parapair_ids.append(paraids[i] + '_' + paraids[j])
        X_batch.append(X_cats)
    X_batch = np.array(X_batch)
    stats = [np.mean(num_paras), np.std(num_paras)/np.sqrt(len(batch_queries)), np.mean(num_k),
             np.std(num_k)/np.sqrt(len(batch_queries)), np.mean(mean_para_per_k),
             np.std(mean_para_per_k)/np.sqrt(len(batch_queries))]
    return torch.from_numpy(X_batch).float(), stats

def prepare_batch_scaled(X_data_batch, emb_size):
    batch_queries = list(X_data_batch.keys())
    X_batch = []
    true_batch_labels = []
    max_paralist_len = max([len(X_data_batch[q]) - 1 for q in batch_queries])
    num_paras = []
    num_k = []
    mean_para_per_k = []
    qvecs = []
    for q in batch_queries:
        #print(q)
        true_batch_label = [d[2] for d in X_data_batch[q][1:]] + [-1] * (max_paralist_len - len(X_data_batch[q][1:]))
        num_clusters = len(list(set(true_batch_label)))
        num_paras.append(len(X_data_batch[q][1:]))
        num_k.append(num_clusters)
        mean_para_per_k.append(len(X_data_batch[q][1:])/num_clusters)
        true_batch_labels.append(true_batch_label)
        qvecs.append(X_data_batch[q][0])
        paraids = []
        paravecs = []
        paravec_labels = []
        for d in X_data_batch[q][1:]:
            paraids.append(d[0])
            paravecs.append(d[1])
            paravec_labels.append(d[2])
        # print(q + " passage count: " + str(len(paraids)))
        for i in range(max_paralist_len - len(X_data_batch[q][1:])):
            paraids.append("dummy"+str(i))
            paravecs.append(np.zeros(emb_size))
            #paravecs.append(np.full(emb_size, -99.0))
            paravec_labels.append(-1)
        X_batch.append(paravecs)
    qvecs = np.array(qvecs)
    num_k = np.array(num_k)
    true_batch_labels = np.array(true_batch_labels)
    true_adj_mat = torch.zeros((true_batch_labels.shape[0], true_batch_labels.shape[1], true_batch_labels.shape[1]))
    for b in range(true_batch_labels.shape[0]):
        for i in range(true_batch_labels.shape[1]):
            for j in range(true_batch_labels.shape[1]):
                if true_batch_labels[b][i] == true_batch_labels[b][j]:
                    true_adj_mat[b][i][j] = 1.0
    X_batch = np.array(X_batch)
    stats = [np.mean(num_paras), np.std(num_paras) / np.sqrt(len(batch_queries)), np.mean(num_k),
             np.std(num_k) / np.sqrt(len(batch_queries)), np.mean(mean_para_per_k),
             np.std(mean_para_per_k) / np.sqrt(len(batch_queries))]
    return torch.from_numpy(X_batch).float(), torch.from_numpy(qvecs).float(), torch.from_numpy(num_k).float(), \
           torch.from_numpy(true_batch_labels).float(), true_adj_mat.float(), stats


def calculate_avg_rand(cand_labels, true_labels):
    avg_rand = 0
    for i in range(len(cand_labels)):
        gt = [l for l in true_labels[i] if l > -0.5]
        cand = [l for l in cand_labels[i][:len(gt)]]
        avg_rand += adjusted_rand_score(gt, cand)
    avg_rand = avg_rand / len(cand_labels)
    return avg_rand

def true_cluster_labels(query_list, X_data):
    max_paralist_len = max([len(X_data[q]) - 1 for q in query_list])
    true_paired_clusters = np.zeros((len(query_list), max_paralist_len, max_paralist_len))
    mask = np.zeros((len(query_list), max_paralist_len, max_paralist_len))
    true_labels = []
    for b in range(len(query_list)):
        q = query_list[b]
        num_paras = len(X_data[q][1:])
        true_batch_label = [d[2] for d in X_data[q][1:]] + [-1] * (max_paralist_len - len(X_data[q][1:]))
        true_labels.append(true_batch_label)
        for i in range(max_paralist_len):
            for j in range(max_paralist_len):
                if i == j or true_batch_label[i] == true_batch_label[j]:
                    true_paired_clusters[b][i][j] = 1.0
                if i < num_paras and j < num_paras:
                    mask[b][i][j] = 1.0
    return torch.from_numpy(true_paired_clusters).float(), torch.from_numpy(np.array(true_labels)).float(), torch.from_numpy(mask).float()

# This function belongs to blackbox-backprop repo

def maybe_parallelize(function, arg_list):
    """
    Parallelizes execution is ray is enabled
    :param function: callable
    :param arg_list: list of function arguments (one for each execution)
    :return:
    """
    # Passive ray module check
    if 'ray' in sys.modules and ray.is_initialized():
        ray_fn = ray.remote(function)
        return ray.get([ray_fn.remote(arg) for arg in arg_list])
    else:
        return [function(arg) for arg in arg_list]