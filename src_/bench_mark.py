import struct
from numpy.random import default_rng
rng = default_rng()
import os
from tqdm import tqdm
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import time  # <-- import time module
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
import numpy as np

DOWNLOAD_SPEED_BPS = 1_000_000_000  # 100 Mbps simulated download speed

def load_vector_bin(file_path):
    vec_dim = 384
    record_size = 4 + 2 + vec_dim * 2
    data = []
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(record_size)
            if not chunk or len(chunk) < record_size:
                break
            code = struct.unpack('<i', chunk[:4])[0]
            page = struct.unpack('<h', chunk[4:6])[0]
            vector = np.frombuffer(chunk[6:], dtype=np.float16).astype(np.float32)
            data.append((code, page, vector))
    return data

def get_query_set():
    all_data = []
    for filename in os.listdir("vectors_save"):
        if filename.endswith(".bin"):
            all_data.extend(load_vector_bin(os.path.join("vectors_save", filename)))

    all_data = all_data
    print(f"Loaded {len(all_data)} vectors total.")
    image_ids = [(code, page) for code, page, _ in all_data]
    vectors = np.stack([vec for _, _, vec in all_data])
    query_indices = rng.choice(len(vectors), size=100, replace=False)
    return vectors, image_ids, query_indices

def simulate_download_delay(num_vectors, vector_dim):
    vector_size_bytes = vector_dim * 4  # float32 = 4 bytes
    total_bytes = num_vectors * vector_size_bytes
    bytes_per_sec = DOWNLOAD_SPEED_BPS // 8
    download_time = total_bytes / bytes_per_sec
    time.sleep(download_time)

def benchmark(vectors, image_ids, query_indices, search_fn, top_k=5):
    recall_at_k = 0
    total = len(query_indices)

    start_time = time.time()  # start timer for benchmark
    for idx in tqdm(query_indices, desc=f"Evaluating Recall@{top_k}"):
        query_vector = vectors[idx]
        query_vector += rng.normal(0, 1e-3, query_vector.shape)
        query_vector /= np.linalg.norm(query_vector)
        true_id = image_ids[idx]

        retrieved_indices = search_fn(query_vector, top_k)
        retrieved_ids = [image_ids[i] for i in retrieved_indices]

        if true_id in retrieved_ids:
            recall_at_k += 1
    elapsed = time.time() - start_time
    recall = recall_at_k / total
    print(f"Recall@{top_k}: {recall:.4f} (Time: {elapsed:.2f} seconds)")
    return recall


def build_faiss_ivf_index(vectors, n_clusters=100, n_probe=5):
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)

    print("Training IVF index...")
    start_time = time.time()
    index.train(vectors)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")

    print("Adding vectors to IVF index...")
    start_time = time.time()
    index.add(vectors)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.2f} seconds")

    index.nprobe = n_probe
    return index


import faiss
import numpy as np
import time

import faiss
import numpy as np
import time

def build_faiss_ivf_index_sampled(
    vectors, n_clusters=100, n_probe=5,
    sample_size=500_000, batch_size=100_000
):
    """
    Args:
        vectors: np.ndarray or memmap of shape (N, d), dtype float32
        n_clusters: number of IVF clusters
        n_probe: number of clusters to search during queries
        sample_size: number of vectors to use for training (sampling from vectors)
        batch_size: number of vectors to add at once to index (to limit memory use)

    Returns:
        index: trained FAISS IVF index
        cluster_sizes: numpy array of shape (n_clusters,), counts of points per cluster
    """
    N, d = vectors.shape

    # Sample vectors for training
    if sample_size < N:
        sample_indices = np.random.choice(N, size=sample_size, replace=False)
        train_vectors = vectors[sample_indices]
    else:
        train_vectors = vectors

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)

    print("Training IVF index on sample...")
    start_time = time.time()
    index.train(train_vectors)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")

    # Assign clusters by searching for nearest centroid (quantizer) on all vectors batch-wise
    cluster_sizes = np.zeros(n_clusters, dtype=np.int64)
    print("Assigning clusters for size tracking...")
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = vectors[start:end]
        # Search nearest cluster centroid for each vector (k=1)
        D, I = index.quantizer.search(batch, 1)
        cluster_ids = I[:, 0]
        # Update counts
        for cid in cluster_ids:
            cluster_sizes[cid] += 1
        print(f"Assigned clusters for vectors {start} to {end}")
    
    for c in sorted(cluster_sizes):
        print(c)
    print("Adding vectors in batches...")
    start_time = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = vectors[start:end]
        index.add(batch)
        print(f"Added vectors {start} to {end}")
    add_time = time.time() - start_time
    print(f"Total add time: {add_time:.2f} seconds")

    index.nprobe = n_probe
    return index, cluster_sizes


def build_faiss_ivf_hnsw_index(vectors, n_clusters=100, n_probe=5, hnsw_m=32, hnsw_ef_construction=40):
    d = vectors.shape[1]

    quantizer = faiss.IndexHNSWFlat(d, hnsw_m)
    quantizer.hnsw.efConstruction = hnsw_ef_construction

    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)

    print("Training IVF+HNSW index...")
    start_time = time.time()
    index.train(vectors)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f} seconds")

    print("Adding vectors to IVF+HNSW index...")
    start_time = time.time()
    index.add(vectors)
    add_time = time.time() - start_time
    print(f"Add time: {add_time:.2f} seconds")

    index.nprobe = n_probe
    return index

def faiss_ivf_search_with_download_sim(index, query_vector, top_k, bandwidth_mbps=100):
    query_vector = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)

    nprobe = index.nprobe
    total_vectors = index.ntotal
    n_clusters = index.nlist
    vector_dim = query_vector.shape[1]

    # Download centroids (quantizer)
    centroids_size_bytes = n_clusters * vector_dim * 4

    # Download vectors in probed clusters
    avg_cluster_size = total_vectors / n_clusters
    vectors_accessed = int(nprobe * avg_cluster_size)
    vectors_size_bytes = vectors_accessed * vector_dim * 4

    total_download_bytes = centroids_size_bytes + vectors_size_bytes

    bytes_per_sec = bandwidth_mbps * 125_000
    delay_sec = total_download_bytes / bytes_per_sec

    time.sleep(delay_sec)
    return indices[0]


def faiss_ivf_hnsw_search_with_download_sim(index, query_vector, top_k, bandwidth_mbps=100):
    query_vector = query_vector.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)

    nprobe = index.nprobe
    total_vectors = index.ntotal
    n_clusters = index.nlist
    vector_dim = query_vector.shape[1]

    # Approximate nodes visited in HNSW quantizer
    try:
        efSearch = index.quantizer.hnsw.efSearch
    except AttributeError:
        efSearch = 40

    centroids_size_bytes = efSearch * vector_dim * 4  # Only visited nodes downloaded

    avg_cluster_size = total_vectors / n_clusters
    vectors_accessed = int(nprobe * avg_cluster_size)
    vectors_size_bytes = vectors_accessed * vector_dim * 4

    total_download_bytes = centroids_size_bytes + vectors_size_bytes

    bytes_per_sec = bandwidth_mbps * 125_000
    delay_sec = total_download_bytes / bytes_per_sec

    time.sleep(delay_sec)
    return indices[0]



def build_meta_clustering(all_vectors, n_clusters1, n_clusters2):
    # Level 1 clustering
    kmeans1 = MiniBatchKMeans(n_clusters=n_clusters1, random_state=42, n_init='auto')
    assignments1 = kmeans1.fit_predict(all_vectors)
    centroids1 = kmeans1.cluster_centers_

    inverted_index_level1 = defaultdict(list)
    for i, c1 in enumerate(assignments1):
        inverted_index_level1[c1].append(i)

    # Prepare for global level2 indexing
    centroids2_list = []
    inverted_index_level2 = defaultdict(list)  # level2_id → list of level1_ids
    level1_to_level2 = {}

    global_level2_id = 0
    for c1_id in range(n_clusters1):
        idxs = inverted_index_level1[c1_id]
        if not idxs:
            continue
        subvecs = all_vectors[idxs]
        n2 = min(n_clusters2, len(subvecs))
        kmeans2 = MiniBatchKMeans(n_clusters=n2, random_state=42, n_init='auto')
        kmeans2.fit(subvecs)

        for local_id in range(n2):
            centroids2_list.append(kmeans2.cluster_centers_[local_id])
            inverted_index_level2[global_level2_id].append(c1_id)
            level1_to_level2[(c1_id, local_id)] = global_level2_id
            global_level2_id += 1

    centroids2 = np.array(centroids2_list)
    return centroids2, centroids1, inverted_index_level1, inverted_index_level2



def my_index_search_with_download_sim(centroids1, centroids2, inverted_index_level1, inverted_index_level2,
                                      all_vectors, query_vector, top_k, bandwidth_mbps=100,
                                      nprobe_level2=1, nprobe_level1=5):
    vector_dim = query_vector.shape[0]

    # 1) Find nearest level2 clusters (on centroids2)
    dists2 = np.linalg.norm(centroids2 - query_vector, axis=1)
    nearest_level2_clusters = np.argsort(dists2)[:nprobe_level2]

    # Download level2 centroids
    centroids2_size_bytes = centroids2.shape[0] * vector_dim * 4
    # Simulate download delay for level2 centroids
    bytes_per_sec = bandwidth_mbps * 125_000
    time.sleep(centroids2_size_bytes / bytes_per_sec)

    candidate_indices = []

    # For each selected level2 cluster, find nearest level1 clusters inside it
    for level2_cluster_id in nearest_level2_clusters:
        level1_clusters_in_level2 = inverted_index_level2[level2_cluster_id]
        # Download those level1 centroids
        centroids1_level2 = centroids1[level1_clusters_in_level2]

        centroids1_size_bytes = len(level1_clusters_in_level2) * vector_dim * 4
        time.sleep(centroids1_size_bytes / bytes_per_sec)

        # Find nearest nprobe_level1 level1 clusters inside this level2 cluster
        dists1 = np.linalg.norm(centroids1_level2 - query_vector, axis=1)
        nearest_level1_indices = np.argsort(dists1)[:nprobe_level1]

        # For each nearest level1 cluster, get vector indices, simulate download
        for idx in nearest_level1_indices:
            cluster1_id = level1_clusters_in_level2[idx]
            vec_indices = inverted_index_level1[cluster1_id]

            # Simulate download of vectors in this cluster
            vectors_size_bytes = len(vec_indices) * vector_dim * 4
            time.sleep(vectors_size_bytes / bytes_per_sec)

            candidate_indices.extend(vec_indices)

    candidate_indices = list(set(candidate_indices))  # unique candidates
    if len(candidate_indices) == 0:
        return []

    candidate_vectors = all_vectors[candidate_indices]

    # Compute similarity and get top_k
    similarities = cosine_similarity(query_vector.reshape(1, -1), candidate_vectors).flatten()
    top_local_indices = np.argsort(similarities)[-top_k:][::-1]

    return [candidate_indices[i] for i in top_local_indices]


import numpy as np


def limit_cluster_sizes(all_vectors, labels, centroids, max_points_per_centroid):
    """
    Reassign points so that no cluster exceeds max_points_per_centroid.

    Params:
        all_vectors: np.ndarray, shape (num_points, dim)
        labels: np.ndarray[int], shape (num_points,)
        centroids: np.ndarray, shape (n_clusters, dim)
        max_points_per_centroid: int, max points allowed per cluster

    Returns:
        new_labels: np.ndarray[int], shape (num_points,)
    """
    n_clusters = centroids.shape[0]
    distances = np.linalg.norm(
        all_vectors[:, None, :] - centroids[None, :, :], axis=2
    )  # shape (num_points, n_clusters)

    final_labels = np.full_like(labels, fill_value=-1)
    assigned_indices_per_cluster = {i: [] for i in range(n_clusters)}

    # Assign closest max_points_per_centroid points per cluster first
    for cluster_id in range(n_clusters):
        cluster_points = np.where(labels == cluster_id)[0]
        if len(cluster_points) <= max_points_per_centroid:
            for idx in cluster_points:
                final_labels[idx] = cluster_id
            assigned_indices_per_cluster[cluster_id] = list(cluster_points)
        else:
            cluster_dists = distances[cluster_points, cluster_id]
            sorted_idx = cluster_points[np.argsort(cluster_dists)]
            chosen = sorted_idx[:max_points_per_centroid]
            overflow = sorted_idx[max_points_per_centroid:]

            for idx in chosen:
                final_labels[idx] = cluster_id
            assigned_indices_per_cluster[cluster_id] = list(chosen)

    # Reassign overflow points to next best cluster with capacity
    overflow_points = np.where(final_labels == -1)[0]
    max_reassign_iters = n_clusters
    for _ in range(max_reassign_iters):
        if len(overflow_points) == 0:
            break

        new_overflow_points = []
        for idx in overflow_points:
            sorted_clusters = np.argsort(distances[idx])
            assigned = False
            for c in sorted_clusters:
                if len(assigned_indices_per_cluster[c]) < max_points_per_centroid:
                    final_labels[idx] = c
                    assigned_indices_per_cluster[c].append(idx)
                    assigned = True
                    break
            if not assigned:
                new_overflow_points.append(idx)

        if len(new_overflow_points) == len(overflow_points):
            print("Warning: Could not assign all overflow points within capacity limits.")
            break
        overflow_points = new_overflow_points

    # For any still unassigned points, assign to nearest cluster forcibly
    unassigned = np.where(final_labels == -1)[0]
    for idx in unassigned:
        nearest = np.argmin(distances[idx])
        final_labels[idx] = nearest
        assigned_indices_per_cluster[nearest].append(idx)

    return final_labels


if __name__ == "__main__":
    all_vectors, image_ids, query_indices = get_query_set()
    globals()["all_vectors"] = all_vectors

    # print("Running brute force benchmark...")
    # benchmark(all_vectors, image_ids, query_indices, brute_force_search, top_k=1)

    print(np.sqrt(len(all_vectors)))
    print("Building FAISS IVF index...")
    faiss_index = build_faiss_ivf_index_sampled(all_vectors, n_clusters=2500, n_probe=1)
    #
    print("Running FAISS IVF benchmark...")
    def faiss_ivf_wrapper(query_vector, top_k):
        return faiss_ivf_search_with_download_sim(faiss_index, query_vector, top_k,bandwidth_mbps=1_000)
    benchmark(all_vectors, image_ids, query_indices, faiss_ivf_wrapper, top_k=1)
