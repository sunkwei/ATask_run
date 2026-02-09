#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)

import numpy as np

from sklearn.cluster._kmeans import k_means
from .infomap_cluster import pred_cluster
from collections import Counter

class SpectralCluster:
    r"""A spectral clustering mehtod using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, num_spks=None, min_num_spks=1, max_num_spks=20, p=0.01):
        self.num_spks = num_spks
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.p = p

    def cosine_similarity(self, M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        return 0.5 * (1.0 + np.dot(M, M.T))

    def prune(self, M):
        m = M.shape[0]
        if m < 1000:
            n = max(m - 10, 2)
        else:
            n = int((1.0 - self.p) * m)
        for i in range(m):
            # high_indexes = np.where(M[i, :] >0.78)[0]
            # low_indexes = np.where(M[i, :] <=0.78)[0]
            indexes = np.argsort(M[i, :])
            low_indexes, high_indexes = indexes[0:n], indexes[n:m]
            M[i, low_indexes] = 0.0
            M[i, high_indexes] = 1.0
        return 0.5 * (M + M.T)

    def laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0.0
        D = np.diag(np.sum(np.abs(M), axis=1))
        return D - M

    def spectral(self, M):
        eig_values, eig_vectors = np.linalg.eigh(M)
        num_spks = self.num_spks if self.num_spks is not None \
            else np.argmax(np.diff(eig_values[:self.max_num_spks + 1])) + 1
        num_spks1 = max(num_spks, self.min_num_spks)
        return eig_vectors[:, :num_spks1], num_spks1, num_spks

    def kmeans(self, emb, k):
        _, labels, _ = k_means(emb, k, random_state=42)
        return labels
    
    def __call__(self, embeddings, min_spk=2):
        # Similarity matrix computation
        self.min_num_spks = min_spk
        # Fallback for trivial cases
        if len(embeddings) < 10:
            return np.zeros(len(embeddings), ).astype(int), min_spk

        # Compute similarity matrix
        similarity_matrix = self.cosine_similarity(embeddings)
        # Prune matrix with p interval
        pruned_similarity_matrix = self.prune(similarity_matrix)
        # Compute Laplacian
        laplacian_matrix = self.laplacian(pruned_similarity_matrix)
        # Compute spectral embeddings
        spectral_embeddings, k, k0 = self.spectral(laplacian_matrix)
        # Assign class labels
        print ("K", k)
        print ("K0", k0)
        
        labels = self.kmeans(spectral_embeddings, k)

        return labels, k0


class ClusterBackend:
    r"""Perfom clustering for input embeddings and output the labels.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self):
        self.model_config = {"merge_thr": 0.78}
        # self.other_config = kwargs

        self.spectral_cluster = SpectralCluster()
        # self.umap_hdbscan_cluster = UmapHdbscan()

    def __call__(self, X):
        # clustering and return the labels
        assert len(X.shape) == 2, "modelscope error: the shape of input should be [N, C]"
        if 0:#X.shape[0] < 20:
            return np.zeros(X.shape[0], dtype="int"), []
        
        labels_spe, k0 = self.spectral_cluster(X)
        CL = dict(Counter(labels_spe))
        CL_num = len(CL)
        if len(X) < 4:
            labels_info = np.zeros(len(X), ).astype(int)
            kn = 4
        else:
            if len(X) < 200:
                kn = 30
            elif CL_num == 2 and max(CL[0], CL[1]) / min(CL[0], CL[1]) >= 10 and max(CL[0], CL[1]) >= 500:
                kn = 200
            else:
                kn = 80
            labels_info = pred_cluster(X, k=kn, min_sim=0.35, CL_num=CL_num)
            LL = []
            for i in range(len(labels_info)):
                LL.append(labels_info[i])
            labels_info = np.array(LL)

            if k0 == 1:
                # 如果谱聚类根据特征聚类是1类，且infomap聚类的最大类占比超过0.8，则重新谱聚类按1类
                counter = Counter(labels_info)
                __, most_num = counter.most_common(1)[0]
                if most_num / len(X) >= 0.8:
                    labels_spe, __ = self.spectral_cluster(X, min_spk=1)
                    
        return labels_spe, labels_info, kn

    def merge_by_cos(self, labels, embs, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            spk_num = labels.max() + 1
            if spk_num == 1:
                break
            spk_center = []
            for i in range(spk_num):
                spk_emb = embs[labels == i].mean(0)
                spk_center.append(spk_emb)
            assert len(spk_center) > 0
            # S = np.ones((len(spk_center), len(spk_center)))
            # for i in range(len(spk_center) - 1):
            #     el0 = spk_center[i]
            #     for j in range(i+1, len(spk_center)):
            #         el1 = spk_center[j]
            #         s = np.mean(np.dot(el0, el1.T))
            #         S[i,j] = s
            spk_center = np.stack(spk_center, axis=0)
            norm_spk_center = spk_center / np.linalg.norm(spk_center, axis=1, keepdims=True)
            affinity = np.matmul(norm_spk_center, norm_spk_center.T)
            affinity = np.triu(affinity, 1)

            spks = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[spks] < cos_thr:
                break
            for i in range(len(labels)):
                if labels[i] == spks[1]:
                    labels[i] = spks[0]
                elif labels[i] > spks[1]:
                    labels[i] -= 1
        return labels
