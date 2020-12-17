from models import cluster
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
import argparse
from hashlib import sha1
random.seed(42)
torch.manual_seed(42)
from numpy.random import seed
seed(42)

def clustering_loss(true_label_batch, cand_label_batch, k=5):
    loss = 0
    for i in range(len(true_label_batch)):
        loss += 1 - np.exp(k * adjusted_rand_score(true_label_batch[i], cand_label_batch[i]))
    return loss/len(true_label_batch)

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

def build_data(page_paras, qrels, paravec_dict, qvec_dict):
    rev_para_label_dict = {}
    selected_pages = [p for p in page_paras.keys() if 'Query:' + sha1(str.encode(p)).hexdigest() in qvec_dict.keys() and
                   len(page_paras[p]) > 9]
    print("Total "+str(len(selected_pages))+" pages")
    with open(qrels, 'r') as q:
        for l in q:
            rev_para_label_dict[l.split(' ')[2]] = l.split(' ')[0]
    X_data = {}
    sorted_pages = sorted(selected_pages, key=lambda k:len(page_paras[k]))
    for page in sorted_pages:
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        qvec = qvec_dict[qid]
        X_data[page] = [qvec]
        labels = list(set([rev_para_label_dict[p] for p in page_paras[page]]))
        for p in page_paras[page]:
            if p not in paravec_dict.keys():
                print(p + " not in para embedding data, skipping...")
                continue
            pvec = paravec_dict[p]
            plabel = labels.index(rev_para_label_dict[p])
            X_data[page].append((p, pvec, plabel))
        # print(page)
    return X_data

def calculate_avg_rand(cand_labels, true_labels):
    avg_rand = 0
    for i in range(len(cand_labels)):
        gt = [l for l in true_labels[i]]
        cand = [l for l in cand_labels[i]]
        avg_rand += adjusted_rand_score(gt, cand)
    avg_rand = avg_rand / len(cand_labels)
    return avg_rand

def true_cluster_labels(query_list, X_data):
    max_paralist_len = max([len(X_data[q]) - 1 for q in query_list])
    true_paired_clusters = np.zeros((len(query_list), max_paralist_len, max_paralist_len))
    true_labels = []
    for b in range(len(query_list)):
        q = query_list[b]
        true_batch_label = [d[2] for d in X_data[q][1:]] + [-1] * (max_paralist_len - len(X_data[q][1:]))
        true_labels.append(true_batch_label)
        for i in range(max_paralist_len):
            for j in range(max_paralist_len):
                if i == j or true_batch_label[i] == true_batch_label[j]:
                    true_paired_clusters[b][i][j] = 1.0
    return torch.from_numpy(true_paired_clusters).float(), torch.from_numpy(np.array(true_labels)).float()

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
    stats = [np.mean(num_paras), np.std(num_paras), np.mean(num_k), np.std(num_k), np.mean(mean_para_per_k), np.std(mean_para_per_k)]
    return torch.from_numpy(X_batch).float(), stats

def train_cats_cluster(X_train, X_val, X_test, batch_size, epochs, emb_size, lambda_val, lrate):
    # X_train/val/test: The dataset will be of the following format
    # {query: [query_vec, (pid1, para vec 1, label 1), (pid2, para vec 2, label 2), ...]} without padding
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Found CUDA device, Using GPU")
    else:
        device = torch.device('cpu')
        print("CUDA device not found, using CPU")
    query_list = list(X_train.keys())
    val_query_list = list(X_val.keys())
    test_query_list = list(X_test.keys())
    #test_query_list = random.sample(test_query_list, 16)
    X_val_data, val_stats = prepare_batch(val_query_list, X_val, emb_size)
    print("Val data mean para %.3f (%.3f), mean k %.3f (%.3f), mean para/k %.3f (%.3f)" % (val_stats[0], val_stats[1],
                                                                                           val_stats[2], val_stats[3],
                                                                                           val_stats[4], val_stats[5]))
    true_val_paired_clusters, true_val_labels = true_cluster_labels(val_query_list, X_val)
    # true_val_labels = true_val_labels.to(device)
    X_test_data, test_stats = prepare_batch(test_query_list, X_test, emb_size)
    print("Test data mean para %.3f (%.3f), mean k %.3f (%.3f), mean para/k %.3f (%.3f)" % (test_stats[0], test_stats[1],
                                                                                           test_stats[2], test_stats[3],
                                                                                           test_stats[4], test_stats[5]))
    true_test_paired_clusters, true_test_labels = true_cluster_labels(test_query_list, X_test)
    # true_test_labels = true_test_labels.to(device)
    m = cluster.CATSCluster(emb_size, lambda_val)
    m = m.to(device)
    #opt = optim.Adam(m.parameters(), lr=lrate)
    opt = optim.RMSprop(m.parameters(), lr=lrate)
    mse_loss = nn.MSELoss().to(device)
    for e in range(epochs):
        print("epoch "+str(e+1)+"/"+str(epochs))
        num_batch = len(query_list)//batch_size + 1
        for b in range(num_batch):
            m = m.to(device)
            m.train()
            opt.zero_grad()
            torch.cuda.empty_cache()
            batch_queries = query_list[b*batch_size:(b+1)*batch_size]
            X_batch, stats = prepare_batch(batch_queries, X_train, emb_size).to(device)
            true_paired_clusters, _ = true_cluster_labels(batch_queries, X_train)
            true_paired_clusters = true_paired_clusters.to(device)
            cand_paired_clusters = m(X_batch).to(device)
            loss = mse_loss(cand_paired_clusters, true_paired_clusters)
            loss.backward()
            opt.step()
            m.cpu()
            m.eval()
            cand_val_paired_clusters = m(X_val_data).detach()
            cand_val_labels = m.predict_cluster_labels().detach()
            val_loss = mse_loss(cand_val_paired_clusters, true_val_paired_clusters)
            print("Batch %d/%d mp %.3f (%.3f), mk %.3f (%.3f), mp/k %.3f (%.3f), Training loss: %.5f, Val loss: %.5f, "
                  "Val avg. AdjRAND: %.5f" % (b+1, num_batch, stats[0], stats[1], stats[2], stats[3], stats[4],
                                              stats[5], loss.item(), val_loss.item(),
                                              calculate_avg_rand(list(cand_val_labels.numpy()),
                                                                 list(true_val_labels.numpy()))))
            if (b+1)%100 == 0:
                num_test = X_test_data.shape[0]
                test_batch_size = 8
                cand_test_paired_clusters = None
                cand_test_labels = None
                for tb in range((num_test // test_batch_size)+1):
                    test_batch = X_test_data[tb*test_batch_size:(tb+1)*test_batch_size,:,:]
                    if cand_test_paired_clusters is None:
                        cand_test_paired_clusters = m(test_batch).detach()
                    else:
                        cand_test_paired_clusters = torch.cat([cand_test_paired_clusters, m(test_batch).detach()], dim=0)
                    if cand_test_labels is None:
                        cand_test_labels = m.predict_cluster_labels().detach()
                    else:
                        cand_test_labels = torch.cat([cand_test_labels, m.predict_cluster_labels().detach()], dim=0)
                test_loss = mse_loss(cand_test_paired_clusters, true_test_paired_clusters).item()
                test_adj_rand = calculate_avg_rand(list(cand_test_labels.numpy()), list(true_test_labels.numpy()))
                print("Test loss: %.5f, Test avg. AdjRAND: %.5f" % (test_loss, test_adj_rand))
    m.cpu()
    m.eval()
    num_test = X_test_data.shape[0]
    test_batch_size = 32
    cand_test_paired_clusters = None
    cand_test_labels = None
    for tb in range((num_test // test_batch_size) + 1):
        test_batch = X_test_data[tb * test_batch_size:(tb + 1) * test_batch_size, :, :]
        if cand_test_paired_clusters is None:
            cand_test_paired_clusters = m(test_batch).detach()
        else:
            cand_test_paired_clusters = torch.cat([cand_test_paired_clusters, m(test_batch).detach()], dim=0)
        if cand_test_labels is None:
            cand_test_labels = m.predict_cluster_labels().detach()
        else:
            cand_test_labels = torch.cat([cand_test_labels, m.predict_cluster_labels().detach()], dim=0)
    test_loss = mse_loss(cand_test_paired_clusters, true_test_paired_clusters).item()
    test_adj_rand = calculate_avg_rand(list(cand_test_labels.numpy()), list(true_test_labels.numpy()))
    print("Test loss: %.5f, Test avg. AdjRAND: %.5f" % (test_loss, test_adj_rand))

def main():
    parser = argparse.ArgumentParser(description='Run CATS model')
    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/bb_cluster_data/")

    #'''
    parser.add_argument('-qtra', '--train_art_qrels', default="base.train.cbor-article.qrels")
    parser.add_argument('-qtr', '--train_qrels', default="base.train.cbor-toplevel.qrels")
    parser.add_argument('-trv', '--train_pvecs', default="y1train-all-paravec-dict.npy")
    parser.add_argument('-trqv', '--train_qvecs', default="half-y1train-qry-attn-context-leadpara-qdict.npy")
    '''
    parser.add_argument('-qtra', '--train_art_qrels', default="train.pages.cbor-article.qrels")
    parser.add_argument('-qtr', '--train_qrels', default="train.pages.cbor-toplevel.qrels")
    parser.add_argument('-trv', '--train_pvecs', default="by1train-all-paravec-dict.npy")
    parser.add_argument('-trqv', '--train_qvecs', default="by1train-context-leadpara-qdict.npy")
    '''
    parser.add_argument('-qta', '--test_art_qrels', default="test.pages.cbor-article.qrels")
    parser.add_argument('-qt', '--test_qrels', default="test.pages.cbor-toplevel.qrels")
    parser.add_argument('-tv', '--test_pvecs', default="by1test-all-paravec-dict.npy")
    parser.add_argument('-tqv', '--test_qvecs', default="by1test-context-leadpara-qdict.npy")
    parser.add_argument('-lr', '--lrate', type=float, default=0.00001)
    parser.add_argument('-bt', '--batch', type=int, default=32)
    parser.add_argument('-ep', '--epochs', type=int, default=3)
    parser.add_argument('-emb', '--emb_size', type=int, default=768)
    parser.add_argument('-l', '--lambda_val', type=float, default=5.0)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    dat = args.data_dir
    page_paras = read_art_qrels(dat+args.train_art_qrels)
    val_page_paras = {k: page_paras[k] for k in random.sample(list(page_paras.keys()), 64)} #####
    train_page_paras = {k: page_paras[k] for k in page_paras.keys() if k not in val_page_paras.keys()}
    test_page_paras = read_art_qrels(dat+args.test_art_qrels)
    train_paravec_dict = np.load(dat + args.train_pvecs, allow_pickle=True)[()]
    test_paravec_dict = np.load(dat + args.test_pvecs, allow_pickle=True)[()]
    train_qvec_dict = np.load(dat + args.train_qvecs, allow_pickle=True)[()]
    test_qvec_dict = np.load(dat + args.test_qvecs, allow_pickle=True)[()]
    print("Embedding vectors loaded, going to build data with articles having at least 10 passages")
    X_train = build_data(train_page_paras, dat+args.train_qrels, train_paravec_dict, train_qvec_dict)
    X_val = build_data(val_page_paras, dat + args.train_qrels, train_paravec_dict, train_qvec_dict)
    X_test = build_data(test_page_paras, dat + args.test_qrels, test_paravec_dict, test_qvec_dict) #####
    # X_test = {k:X_test[k] for k in random.sample(X_test.keys(), 16)} #####
    print("Dataset built, going to start training")
    train_cats_cluster(X_train, X_val, X_test, args.batch, args.epochs, args.emb_size, args.lambda_val, args.lrate)


if __name__ == '__main__':
    main()