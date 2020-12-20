from models import cluster, utils
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
import os
import time
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

def build_data(page_paras, qrels, paravec_dict, qvec_dict, nosort, is_test=False, min_num_paras=10, min_nump_k=2.0, max_nump_k=4.0):
    rev_para_label_dict = {}
    with open(qrels, 'r') as q:
        for l in q:
            rev_para_label_dict[l.split(' ')[2]] = l.split(' ')[0]
    selected_pages = []
    if is_test:
        selected_pages = list(page_paras.keys())
    else:
        for p in page_paras.keys():
            num_paras = len(page_paras[p])
            labels = list(set([rev_para_label_dict[p] for p in page_paras[p]]))
            nump_k = num_paras / len(labels)
            if 'Query:' + sha1(str.encode(p)).hexdigest() in qvec_dict.keys() and num_paras >= min_num_paras and \
                    nump_k < max_nump_k and nump_k > min_nump_k:
                selected_pages.append(p)
    if nosort:
        sorted_pages = selected_pages
    else:
        sorted_pages = sorted(selected_pages, key=lambda k: len(page_paras[k]))
    X_data = {}
    for page in sorted_pages:
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        qvec = qvec_dict[qid]
        X_data[page] = [qvec]
        labels = list(set([rev_para_label_dict[p] for p in page_paras[page]]))
        for p in page_paras[page]:
            if p not in paravec_dict.keys():
                print(p + " not in para embedding data from page "+page+", skipping...")
                continue
            pvec = paravec_dict[p]
            plabel = labels.index(rev_para_label_dict[p])
            X_data[page].append((p, pvec, plabel))
        # print(page)
    print("Total " + str(len(sorted_pages)) + " pages")
    return X_data

def train_cats_cluster(X_train, X_val, X_test, batch_size, epochs, emb_size, lambda_val, lrate, save, early_stop_b):
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
    X_val_data, val_stats = utils.prepare_batch(val_query_list, X_val, emb_size)
    print("Val data mean para (serr) %.2f (%.2f), mean k (serr) %.2f (%.2f), mean para/k (serr) %.2f (%.2f)" % (val_stats[0], val_stats[1],
                                                                                           val_stats[2], val_stats[3],
                                                                                           val_stats[4], val_stats[5]))
    true_val_paired_clusters, true_val_labels = utils.true_cluster_labels(val_query_list, X_val)
    # true_val_labels = true_val_labels.to(device)
    X_test_data, test_stats = utils.prepare_batch(test_query_list, X_test, emb_size)
    print("Test data mean para (serr) %.2f (%.2f), mean k (serr) %.2f (%.2f), mean para/k (serr) %.2f (%.2f)" % (test_stats[0], test_stats[1],
                                                                                           test_stats[2], test_stats[3],
                                                                                           test_stats[4], test_stats[5]))
    true_test_paired_clusters, true_test_labels = utils.true_cluster_labels(test_query_list, X_test)
    # true_test_labels = true_test_labels.to(device)
    m = cluster.CATSCluster(emb_size, lambda_val)
    m = m.to(device)
    opt = optim.Adam(m.parameters(), lr=lrate)
    #opt = optim.RMSprop(m.parameters(), lr=lrate)
    mse_loss = nn.MSELoss().to(device)
    early_stopped = False
    init_batch_num = 100
    for e in range(epochs):
        print("epoch "+str(e+1)+"/"+str(epochs))
        num_batch = len(query_list)//batch_size + 1
        for b in range(num_batch):
            m = m.to(device)
            m.train()
            opt.zero_grad()
            batch_queries = query_list[b*batch_size:(b+1)*batch_size]
            X_batch, stats = utils.prepare_batch(batch_queries, X_train, emb_size)
            X_batch = X_batch.to(device)
            true_paired_clusters, _ = utils.true_cluster_labels(batch_queries, X_train)
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
            val_rand = utils.calculate_avg_rand(list(cand_val_labels.numpy()), list(true_val_labels.numpy()))
            print("Batch %d/%d mp(serr) %.2f (%.2f), mk(serr) %.2f (%.2f), mp/k(serr) %.2f (%.2f), Training loss: %.5f, Val loss: %.5f, "
                  "Val avg. AdjRAND: %.5f" % (b+1, num_batch, stats[0], stats[1], stats[2], stats[3], stats[4],
                                              stats[5], loss.item(), val_loss.item(), val_rand))
            if b > init_batch_num and early_stop_b > 0 and b >= early_stop_b:
                print("Reached at desired epoch/batch num, stopping early...")
                early_stopped = True
                break
            '''
            if (b+1)%100 == 0:
                cand_test_paired_clusters = m(X_test_data).detach()
                cand_test_labels = m.predict_cluster_labels().detach()
                test_loss = mse_loss(cand_test_paired_clusters, true_test_paired_clusters).item()
                test_adj_rand = calculate_avg_rand(list(cand_test_labels.numpy()), list(true_test_labels.numpy()))
                print("Test loss: %.5f, Test avg. AdjRAND: %.5f" % (test_loss, test_adj_rand))
            '''
        if early_stopped:
            break
    m.cpu()
    m.eval()
    cand_test_paired_clusters = m(X_test_data).detach()
    cand_test_labels = m.predict_cluster_labels().detach()
    test_loss = mse_loss(cand_test_paired_clusters, true_test_paired_clusters).item()
    test_adj_rand = utils.calculate_avg_rand(list(cand_test_labels.numpy()), list(true_test_labels.numpy()))
    print("Test loss: %.5f, Test avg. AdjRAND: %.5f" % (test_loss, test_adj_rand))
    if save:
        if not os.path.isdir('saved_models'):
            os.makedirs('saved_models')
        torch.save(m.state_dict(), 'saved_models/'+time.strftime('%b-%d-%Y_%H%M', time.localtime())+'.model')

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
    parser.add_argument('-vb', '--val_size', type=int, default=32)
    parser.add_argument('-ep', '--epochs', type=int, default=3)
    parser.add_argument('-emb', '--emb_size', type=int, default=768)
    parser.add_argument('-l', '--lambda_val', type=float, default=100.0)
    parser.add_argument('-es', '--early_stop_b', type=int, default=-1)
    parser.add_argument('--nosort', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    dat = args.data_dir
    page_paras = utils.read_art_qrels(dat+args.train_art_qrels)
    #val_page_paras = {k: page_paras[k] for k in random.sample(list(page_paras.keys()), 64)} #####
    #train_page_paras = {k: page_paras[k] for k in page_paras.keys() if k not in val_page_paras.keys()}
    test_page_paras = utils.read_art_qrels(dat+args.test_art_qrels)
    train_paravec_dict = np.load(dat + args.train_pvecs, allow_pickle=True)[()]
    test_paravec_dict = np.load(dat + args.test_pvecs, allow_pickle=True)[()]
    train_qvec_dict = np.load(dat + args.train_qvecs, allow_pickle=True)[()]
    test_qvec_dict = np.load(dat + args.test_qvecs, allow_pickle=True)[()]
    print("Embedding vectors loaded, going to build data with articles having at least 10 passages")
    X_data = build_data(page_paras, dat+args.train_qrels, train_paravec_dict, train_qvec_dict, args.nosort)
    X_val = {k:X_data[k] for k in random.sample(list(X_data.keys()), args.val_size)}
    X_train = {k:X_data[k] for k in X_data.keys() if k not in X_val.keys()}
    X_test = build_data(test_page_paras, dat + args.test_qrels, test_paravec_dict, test_qvec_dict, args.nosort, True) #####
    #X_test = {k:X_test[k] for k in random.sample(X_test.keys(), 32)} #####
    print("Dataset built, going to start training")
    train_cats_cluster(X_train, X_val, X_test, args.batch, args.epochs, args.emb_size, args.lambda_val, args.lrate,
                       args.save, args.early_stop_b)


if __name__ == '__main__':
    main()