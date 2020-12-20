from models import cluster, utils
import torch
from hashlib import sha1
import numpy as np
import json
import argparse

def build_data(page_paras, qrels, paravec_dict, qvec_dict, nosort):
    rev_para_label_dict = {}
    with open(qrels, 'r') as q:
        for l in q:
            rev_para_label_dict[l.split(' ')[2]] = l.split(' ')[0]
    selected_pages = list(page_paras.keys())
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

def save_cluster_labels(model_path, cluster_label_path, art_qrels, qrels, paravec_file, qvec_file, emb_size, lambda_val):
    paravec_dict = np.load(paravec_file, allow_pickle=True)[()]
    qvec_dict = np.load(qvec_file, allow_pickle=True)[()]
    page_paras = utils.read_art_qrels(art_qrels)
    X_data = build_data(page_paras, qrels, paravec_dict, qvec_dict, True)
    query_list = list(X_data.keys())
    X, stats = utils.prepare_batch(query_list, X_data, emb_size)
    model = cluster.CATSCluster(emb_size, lambda_val)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    cand_test_paired_clusters = model(X)
    cand_labels = model.predict_cluster_labels()
    true_paired_clusters, true_labels = utils.true_cluster_labels(query_list, X_data)
    adj_rand = utils.calculate_avg_rand(list(cand_labels.numpy()), list(true_labels.numpy()))
    print("Avg. AdjRAND: %.5f" % (adj_rand))
    with open(cluster_label_path, 'w') as f:
        json.dump({'X_data':X_data, 'labels':cand_labels.tolist()}, f)

def main():
    parser = argparse.ArgumentParser(description='Save cluster labels')
    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/bb_cluster_data/")
    parser.add_argument('-qra', '--art_qrels', default="test.pages.cbor-article.qrels")
    parser.add_argument('-qr', '--qrels', default="test.pages.cbor-toplevel.qrels")
    parser.add_argument('-pv', '--pvecs', default="by1test-all-paravec-dict.npy")
    parser.add_argument('-qv', '--qvecs', default="by1test-context-leadpara-qdict.npy")
    parser.add_argument('-mp', '--model_path', default="/home/sk1105/sumanta/BB_cluster/saved_models/lambda100_bestval.model")
    parser.add_argument('-emb', '--emb_size', type=int, default=768)
    parser.add_argument('-l', '--lambda_val', type=float, default=100.0)
    parser.add_argument('-op', '--outpath', default="/home/sk1105/sumanta/BB_cluster/saved_models/lambda100_bestval.model.cluster-labels.json")

    args = parser.parse_args()
    dat = args.data_dir
    save_cluster_labels(args.model_path, args.outpath, dat+args.art_qrels, dat+args.qrels, dat+args.pvecs,
                        dat+args.qvecs, args.emb_size, args.lambda_val)

if __name__ == '__main__':
    main()