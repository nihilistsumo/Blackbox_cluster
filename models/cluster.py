import torch
import torch.nn as nn
from .utils import maybe_parallelize
import numpy as np
import math
from sklearn.cluster import AgglomerativeClustering

def clustering(batch_pairscore_matrix, num_clusters):
    cluster_labels = []
    batch_pairscore_matrix = batch_pairscore_matrix.detach().cpu()
    for i in range(batch_pairscore_matrix.shape[0]):
        clustering_algo = AgglomerativeClustering(n_clusters=int(num_clusters[i].item()), affinity='precomputed', linkage='average')
        cluster_labels.append(clustering_algo.fit_predict(batch_pairscore_matrix[i]))
    cluster_labels = np.array(cluster_labels)

    num_batch = batch_pairscore_matrix.shape[0]
    p = batch_pairscore_matrix.shape[1]
    paired_cluster_matrices = np.zeros(batch_pairscore_matrix.shape)
    for b in range(num_batch):
        cand_label = cluster_labels[b]
        for i in range(p):
            for j in range(p):
                if i == j or cand_label[i] == cand_label[j]:
                    paired_cluster_matrices[b][i][j] = 1.0
    return paired_cluster_matrices, cluster_labels

class OptimCluster(torch.autograd.Function):

    @staticmethod
    def forward(ctx, batch_pairscore_matrix, lambda_val, num_clusters):
        """
        :param ctx: context for backpropagation
        :param dbscan_clusterer: sklearn clustering instance, c = sklearn.DBSCAN(eps=eps, min_samples=min_samples)
        :param pairscores: (torch.Tensor of shape [maxlen C 2]): parapair scores
        :param lambda_val: hyperparam lambda
        :return: cluster labels (torch.Tensor of shape [n paras])
        """
        ctx.lambda_val = lambda_val
        ctx.num_clusters = num_clusters
        ctx.batch_pairscore_matrix = batch_pairscore_matrix
            # ctx.cluster_labels = np.array(maybe_parallelize(cluster_pairscores, arg_list=[ctx.pairscore_matrix, clustering]))
            # cluster_labels.append(cluster_pairscores(pairscore_matrix, num_clusters))
        ctx.paired_cluster_matrices, _ = clustering(ctx.batch_pairscore_matrix, num_clusters)
        return torch.from_numpy(ctx.paired_cluster_matrices).float().to(ctx.batch_pairscore_matrix.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_numpy = grad_output.detach().cpu().numpy()
        batch_pairscore_matrix_numpy = ctx.batch_pairscore_matrix.detach().cpu().numpy()
        batch_pairscore_matrix_prime = torch.from_numpy(np.maximum(
            batch_pairscore_matrix_numpy + ctx.lambda_val * grad_output_numpy, 0.0)).to(ctx.batch_pairscore_matrix.device)
        better_paired_cluster_matrices, _ = clustering(batch_pairscore_matrix_prime, ctx.num_clusters)
        gradient = -(ctx.paired_cluster_matrices - better_paired_cluster_matrices) / ctx.lambda_val
        return torch.from_numpy(gradient).to(ctx.batch_pairscore_matrix.device), None, None

class CATS(nn.Module):
    def __init__(self, emb_size):
        super(CATS, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, emb_size)
        self.LL2 = nn.Linear(emb_size, emb_size)
        self.LL3 = nn.Linear(5 * emb_size, 1)

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (n X mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (n X mC2)
        '''
        self.Xq = X[:, :, :self.emb_size]
        self.Xp1 = X[:, :, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, :, 2 * self.emb_size:]
        self.z1 = torch.abs(self.Xp1 - self.Xq)
        self.z2 = torch.abs(self.Xp2 - self.Xq)
        self.zdiff = torch.abs(self.Xp1 - self.Xp2)
        self.zp1 = torch.relu(self.LL2(torch.relu(self.LL1(self.Xp1))))
        self.zp2 = torch.relu(self.LL2(torch.relu(self.LL1(self.Xp2))))
        self.zql = torch.relu(self.LL2(torch.relu(self.LL1(self.Xq))))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.z = torch.cat((self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=2)
        o = torch.relu(self.LL3(self.z))
        o = o.reshape((-1, o.shape[1]))
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        y_pred = self.forward(X_test)
        return y_pred

class QuerySiamese(nn.Module):
    def __init__(self, emb_size):
        super(QuerySiamese, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, 256)
        self.LL2 = nn.Linear(256, 64)
        self.LL3 = nn.Linear(emb_size, 256)
        self.LL4 = nn.Linear(256, 64)

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (n X mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (n X mC2)
        '''
        self.Xq = X[:, :, :self.emb_size]
        self.Xp1 = X[:, :, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, :, 2 * self.emb_size:]
        self.Xp1z = torch.relu(self.LL2(torch.relu(self.LL1(self.Xp1))))
        self.Xp2z = torch.relu(self.LL2(torch.relu(self.LL1(self.Xp2))))
        self.Xqz = torch.relu(self.LL4(torch.relu(self.LL3(self.Xp1))))
        self.Xqp1z = torch.einsum('bmi,bmj->bmij', (self.Xqz, self.Xp1z)).reshape((X.shape[0], X.shape[1], -1))
        self.Xqp2z = torch.einsum('bmi,bmj->bmij', (self.Xqz, self.Xp2z)).reshape((X.shape[0], X.shape[1], -1))
        self.Xqpz = torch.mean((self.Xqp1z - self.Xqp2z)**2, 2)

        return self.Xqpz

class QueryScaler(nn.Module):
    def __init__(self, emb_size):
        super(QueryScaler, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, 256, bias=False)
        self.LL2 = nn.Linear(256, 128, bias=False)
        self.LL3 = nn.Linear(emb_size, 256, bias=False)
        self.LL4 = nn.Linear(256, 128, bias=False)
        self.LL5 = nn.Linear(128, 1, bias=False)

    def forward(self, X):
        self.Xq = X[:, :, :self.emb_size]
        self.Xp1 = X[:, :, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, :, 2 * self.emb_size:]
        self.Xp1z = torch.relu(self.LL2(torch.relu(self.LL1(self.Xp1))))
        self.Xp2z = torch.relu(self.LL2(torch.relu(self.LL1(self.Xp2))))
        self.Xqz = torch.relu(self.LL4(torch.relu(self.LL3(self.Xq))))
        self.Xpz = torch.abs(self.Xp1z - self.Xp2z)
        self.Xpqz = self.Xqz * self.Xpz
        self.dist = torch.tanh(torch.relu(self.LL5(self.Xpqz)))
        self.dist = self.dist.reshape((-1, self.dist.shape[1]))

        return self.dist

class CATSCluster(nn.Module):
    def __init__(self, emb_size, lambda_val):
        super(CATSCluster, self).__init__()
        self.lambda_val = lambda_val
        self.cats = CATS(emb_size)
        self.qsiam = QuerySiamese(emb_size)
        self.qs = QueryScaler(emb_size)
        self.optim = OptimCluster()

    def forward(self, X_data):
        # X_data is of shape n X maxlen^2 X 3*emb
        # output is of shape n X maxlen X maxlen
        self.num_clusters = X_data[:, 0, 0].reshape(-1)
        self.pairscores = self.qs(X_data[:, 1:, :])
        num_batch = self.pairscores.shape[0]
        maxlen = int(np.sqrt(self.pairscores.shape[1]))
        self.batch_pairscore_matrix = self.pairscores.reshape((num_batch, maxlen, maxlen))
        #print(self.batch_pairscore_matrix[5])
        self.batch_cluster_matrices = self.optim.apply(self.batch_pairscore_matrix, self.lambda_val, self.num_clusters)
        return self.batch_cluster_matrices

    def predict_cluster_labels(self):
        _, self.cluster_labels = clustering(self.batch_pairscore_matrix, self.num_clusters)
        self.cluster_labels = torch.from_numpy(self.cluster_labels).float().to(self.batch_pairscore_matrix.device)
        return self.cluster_labels


    ### This forward accepts X_data of shape n X (mC2) X 3*emb ###
    ### This is suitable for old version of cats that is not invariant to order of q,p1,p2 ###
    ### This is due to the order of concat layer of cats ###
    ### But this is not diff maybe because of for loop ###
    '''
    def forward(self, X_data):
        # X_data is of shape n X (maxlen C 2) X 3*emb
        # output is of shape n X maxlen X maxlen
        self.pairscores = self.cats(X_data)
        num_batch = self.pairscores.shape[0]
        p = round((1 + math.sqrt(1 + 8 * self.pairscores.shape[1])) / 2)
        self.batch_pairscore_matrix = np.zeros((num_batch, p, p))
        for i in range(self.pairscores.shape[0]):
            pairscores = self.pairscores[i]
            pairscore_matrix = np.zeros((p, p))
            c = 0
            for i in range(num_batch - 1):
                for j in range(i + 1, pairscore_matrix.shape[1]):
                    pairscore_matrix[i][j] = pairscore_matrix[j][i] = pairscores[c]
                    c += 1
            self.batch_pairscore_matrix[i] = pairscore_matrix
        self.batch_cluster_matrices = self.optim.apply(self.batch_pairscore_matrix, self.lambda_val, self.num_clusters)
        return self.batch_cluster_matrices
    '''
    '''
    def forward(self, X_data):
        cluster_cand_labels = {}
        for q in batch_queries:
            qvec = X_data[q][0]
            paraids = []
            paravecs = []
            paravec_labels = []
            for d in X_data[q][1:]:
                paraids.append(d[0])
                paravecs.append(d[1])
                paravec_labels.append(d[2])
            # print(q + " passage count: " + str(len(paraids)))
            X_cats = []
            parapair_ids = []
            for i in range(len(paraids)-1):
                for j in range(i+1, len(paraids)):
                    X_cats.append(np.hstack((qvec, paravecs[i], paravecs[j])))
                    parapair_ids.append(paraids[i]+'_'+paraids[j])
            X_cats = torch.tensor(np.array(X_cats))
            parapair_scores = self.cats(X_cats)
            if not torch.sum(parapair_scores).item() == 0:
                parapair_scores = (parapair_scores - torch.min(parapair_scores)) / (torch.max(parapair_scores) - torch.min(parapair_scores))
            parapair_score_dict = {}
            for i in range(len(parapair_ids)):
                parapair_score_dict[parapair_ids[i]] = 1-parapair_scores[i].item()
            pair_score_matrix = []
            for i in range(len(paraids)):
                curr_scores = []
                for j in range(len(paraids)):
                    if i==j:
                        curr_scores.append(0)
                    elif paraids[i]+'_'+paraids[j] in parapair_score_dict.keys():
                        curr_scores.append(parapair_score_dict[paraids[i] + '_' + paraids[j]])
                    else:
                        curr_scores.append(parapair_score_dict[paraids[j] + '_' + paraids[i]])
                pair_score_matrix.append(curr_scores)
            cand_labels = list(self.optim.apply(pair_score_matrix, self.lambda_val, len(set(paravec_labels))).numpy())
            cluster_cand_labels[q] = cand_labels
        return cluster_cand_labels
        '''

class DeepTransformCluster(nn.Module):
    def __init__(self, emb_size, lambda_val):
        super(DeepTransformCluster).__init__()
        self.emb_size = emb_size
        self.lambda_val = lambda_val
        self.transform_mat = torch.zeros((emb_size, emb_size), requires_grad=True)
        self.query_weight = torch.tensor([1.0, 0.0], requires_grad=True)
        self.optim = OptimCluster()

    def forward(self, X_data, qvecs, ks):
        maxlen = X_data.shape[1]
        X_paired_dist_mat = torch.cdist(X_data, X_data, p=2)
        scaling_vecs = torch.matmul(qvecs, self.transform_mat)
        scaling_vecs = scaling_vecs.unsqueeze(1).repeat((1, maxlen, 1))
        X_scaled_dat = X_data * scaling_vecs
        X_scaled_paired_dist_mat = torch.cdist(X_scaled_dat, X_scaled_dat, p=2)
        X_comb_paired_dist_mat = self.query_weight[0] * X_paired_dist_mat + self.query_weight[1] * X_scaled_paired_dist_mat
        self.batch_adj_mat = self.optim.apply(X_comb_paired_dist_mat, self.lambda_val, ks)
        return self.batch_adj_mat