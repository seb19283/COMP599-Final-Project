from collections import defaultdict
import torch
import time
import numpy as np
from torch import Tensor as T
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.cluster import MiniBatchKMeans
from repository import EmbDataset

class Corpora():
    def __init__(
        self,
        doc_embeddings
    ):
        self.doc_embeddings = doc_embeddings
             
    def get_top_k(
        self,
        q_emb,
        doc_embs,
        k
    ):
        doc_ids = []
        if len(doc_embs) == 2: #If it is a tuple, it has embeds and ids
            doc_embs, doc_ids = doc_embs
            doc_ids = [doc_id for sublist in doc_ids for doc_id in sublist]
        doc_embs = torch.cat(doc_embs).to(device)
        sim_search_all = torch.matmul(q_emb, doc_embs.T)
        _, top_k_indices = torch.topk(sim_search_all, k)
        if len(doc_ids) > 0:
            top_k_indices = T([doc_ids[i] for i in top_k_indices]).int().to(device) # get real index, not the one from inside the cluster
        return top_k_indices 
    
    
    def update_doc_embeddings(
        self,
        doc_embeddings: T
    ):
        self.doc_embeddings = doc_embeddings
        
    
    def get_top_k_for_q_embed(
        self,
        q_embeddings: T,
        k
    ):
        raise NotImplementedError('Need to call this method from a subclass object')

class Bruteforce(Corpora):
    def __init__(
        self,
        doc_embeddings
    ):
        super().__init__(doc_embeddings)
            
    def get_top_k_for_q_embed(
        self,
        q_embeddings: T,
        k
    ):
        return self.get_top_k(q_embeddings, [self.doc_embeddings], k)
        

class EvidenceClustering(Corpora):
    def __init__(
        self,
        doc_embeddings,
        k #nb of clusters
    ):
        super().__init__(doc_embeddings)
        self.clustering = None
        self.cluster2ids = defaultdict(list)
        self.cluster2embeds = defaultdict(list)
        self.nb_clusters = k
        self.create_clustering()
    
    
    def cluster_indices(self, clustNum, labels_array):
        return np.where(labels_array == clustNum)[0]
    
    
    def create_clustering(
        self
    ):
        if self.doc_embeddings is None:
            print("Document embeddings have not been created yet. Cannot create clustering.")
            return
        
        print("Start clustering")
        k = self.nb_clusters
        t0 = time.time()
        batch_size = 10 * k
        init_size = 3 * batch_size
        km = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=3,
            init_size=init_size,
            batch_size=batch_size
        )
        km.fit(self.doc_embeddings.cpu())
        
        for i in range(k):
            ids_in_cluster = self.cluster_indices(i, km.labels_)
            nb_ids_in_cluster = len(ids_in_cluster)
            embs_in_cluster = torch.cat([self.doc_embeddings[j] for j in ids_in_cluster])
            embs_in_cluster = embs_in_cluster.reshape(nb_ids_in_cluster, 768)
            self.cluster2ids[i] = ids_in_cluster
            self.cluster2embeds[i] = embs_in_cluster
        self.clustering = km   
        print(f"Clustering took {time.time()-t0}s")
        
    def get_top1_centroids(
        self,
        q_embeddings: T
    ):
        return self.clustering.predict(q_embeddings.cpu())
        
    def get_centroid_embeddings(
        self,
        centroid_ids
    ):
        all_centroid_embeds = T(self.clustering.cluster_centers_).to(device)
        return torch.stack([all_centroid_embeds[i] for i in centroid_ids])
        
    def get_centroids_for_questions(
        self,
        q_embeddings: T
    ):
        return self.get_centroid_embeddings(self.get_top1_centroids(q_embeddings))
        
    def get_centroids(
        self,
        q_embeddings: T,
        k
    ):
        centroids = self.get_top1_centroids(q_embeddings)
        centroids_list = []
        for i in range(len(centroids)):
            centroid = centroids[i]
            top_centroids = []
            top_centroids.append(centroid)
            if len(self.get_ids_from_cluster(centroid)) < k:
                centroids_embeds = T(self.clustering.cluster_centers_)
                nb_centroids = 2
                while sum([len(self.get_ids_from_cluster(top_centroid)) for top_centroid in top_centroids]) < k:  
                    top_centroids = self.get_top_k(q_embeddings[i], [centroids_embeds], nb_centroids).tolist()
                    nb_centroids = nb_centroids + 1
            centroids_list.append(top_centroids)
        return centroids_list
    
    def get_ids_from_cluster(
        self,
        cluster_id
    ):
        return self.cluster2ids[cluster_id]
    
    def get_embeds_from_cluster(
        self,
        cluster_ids
    ):
        embeds = []
        ids = []
        for cluster_id in cluster_ids:
            if len(self.cluster2ids[cluster_id]) == 0 :
                continue
            embeds.append(self.cluster2embeds[cluster_id])
            ids.append(self.cluster2ids[cluster_id])
        return (embeds, ids)
    
    def get_top_k_for_q_embed(
        self,
        q_embeddings: T,
        k
    ):
        centroids = self.get_centroids(q_embeddings, k)  
        doc_embs_list = []
        for i in range(len(centroids)):
            doc_embs_list.append(self.get_top_k(q_embeddings[i], self.get_embeds_from_cluster(centroids[i]), k))
        return torch.stack(doc_embs_list).to(device)
     
        
    def update_doc_embeddings(
        self,
        doc_embeddings: T
    ):
        super().update_doc_embeddings(doc_embeddings)
        self.cluster2ids = defaultdict(list)
        self.cluster2embeds = defaultdict(list)
        
    
    def update_clustering(
        self,
        doc_embeddings: T,
        batch_size
    ):
        self.doc_embeddings = doc_embeddings
        self.cluster2ids = defaultdict(list)
        self.cluster2embeds = defaultdict(list)

        docs_dataset = EmbDataset(doc_embeddings)
        loader = DataLoader(docs_dataset, batch_size=batch_size, shuffle=False)

        i = 0
        for doc_embeds in tqdm(loader):
            best_centroids = self.get_top1_centroids(doc_embeds)
            j = 0
            for doc_embed in doc_embeds:
                self.cluster2ids[best_centroids[j]].append((i*batch_size)+j)
                if len(self.cluster2embeds[best_centroids[j]]) == 0:
                    self.cluster2embeds[best_centroids[j]] = doc_embed
                elif len(self.cluster2embeds[best_centroids[j]]) == 1:
                    self.cluster2embeds[best_centroids[j]] = torch.stack([self.cluster2embeds[best_centroids[j]], doc_embed])
                else:
                    self.cluster2embeds[best_centroids[j]] = torch.cat([self.cluster2embeds[best_centroids[j]], doc_embed])
                j += 1
            i += 1

        for centroid_id, embeds in self.cluster2embeds.items():
            self.cluster2embeds[centroid_id] = embeds.reshape(int((len(embeds)/768)), 768)
