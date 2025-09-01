import struct
import numpy as np
import os
import gzip
import faiss

def save_cluster_bin(file_path, codes, pages, vectors):
    with gzip.open(file_path, "wb") as f:
        for code, page, vec in zip(codes, pages, vectors):
            f.write(struct.pack('<i', code))
            f.write(struct.pack('<h', page))
            f.write(vec.tobytes())  # vec already float16

def load_all_vectors_and_ids(folder="vectors_save", vec_dim=384):
    filenames = [fn for fn in os.listdir(folder) if fn.endswith(".bin")]
    total_records = 0
    # First pass: count total vectors to preallocate
    for fn in filenames:
        filepath = os.path.join(folder, fn)
        filesize = os.path.getsize(filepath)
        record_size = 4 + 2 + vec_dim * 2  # int32 + int16 + float16 vec
        total_records += filesize // record_size

    codes = np.empty(total_records, dtype=np.int32)
    pages = np.empty(total_records, dtype=np.int16)
    vectors = np.empty((total_records, vec_dim), dtype=np.float16)

    idx = 0
    record_size = 4 + 2 + vec_dim * 2
    for fn in filenames:
        filepath = os.path.join(folder, fn)
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(record_size)
                if len(chunk) < record_size:
                    break
                codes[idx] = struct.unpack('<i', chunk[:4])[0]
                pages[idx] = struct.unpack('<h', chunk[4:6])[0]
                vectors[idx] = np.frombuffer(chunk[6:], dtype=np.float16)
                idx += 1

    print(f"Loaded {idx} vectors total.")
    return codes[:idx], pages[:idx], vectors[:idx]

def cluster_and_save(
    codes, pages, vectors,
    output_dir,
    n_clusters=100,
    sample_size=1_000_000,
    batch_size=100_000,
):
    os.makedirs(output_dir, exist_ok=True)

    N, d = vectors.shape

    # Faiss requires float32, so convert sample and batch vectors only on demand
    if sample_size < N:
        sample_indices = np.random.choice(N, size=sample_size, replace=False)
        sample = vectors[sample_indices].astype(np.float32)
    else:
        sample = vectors.astype(np.float32)

    print("Training k-means clustering...")
    clustering = faiss.Clustering(d, n_clusters)
    clustering.niter = 20
    clustering.max_points_per_centroid = 100_000

    index = faiss.IndexFlatL2(d)
    clustering.train(sample, index)

    centroids = faiss.vector_to_array(clustering.centroids).reshape(n_clusters, d)
    # Save centroids in float16 format
    centroids_fp16 = centroids.astype(np.float16)
    with open(os.path.join(output_dir, "centroids.bin"), "wb") as f:
        f.write(centroids_fp16.tobytes())
    print(f"Saved centroids.npy with shape {centroids.shape}")

    # Assign all vectors to clusters in batches (convert to float32 on the fly)
    cluster_ids = np.empty(N, dtype=np.int32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_vecs = vectors[start:end].astype(np.float32)
        _, I = index.search(batch_vecs, 1)
        cluster_ids[start:end] = I[:, 0]
        print(f"Assigned clusters for vectors {start} to {end}")

    print("Saving clusters in binary format...")
    for cid in range(n_clusters):
        indices = np.where(cluster_ids == cid)[0]
        if indices.size == 0:
            print(f"Cluster {cid} is empty, skipping save.")
            continue

        c_codes = codes[indices]
        c_pages = pages[indices]
        c_vecs = vectors[indices]  # already float16

        out_path = os.path.join(output_dir, f"cluster_{cid}.bin.gz")
        save_cluster_bin(out_path, c_codes, c_pages, c_vecs)
        print(f"Saved {out_path} with {len(indices)} vectors")

    return centroids, index, vectors

def benchmark_recall(centroids, index, vectors, n_queries=100, noise_std=0.01):
    N = vectors.shape[0]
    query_indices = np.random.choice(N, size=n_queries, replace=False)
    original_vecs = vectors[query_indices].astype(np.float32)
    noisy_vecs = original_vecs + np.random.normal(scale=noise_std, size=original_vecs.shape).astype(np.float32)

    _, original_assign = index.search(original_vecs, 1)
    _, noisy_assign = index.search(noisy_vecs, 1)

    matches = (original_assign[:, 0] == noisy_assign[:, 0])
    recall = np.mean(matches)
    print(f"Benchmark recall (noisy query cluster matches original): {recall:.4f} over {n_queries} samples")
    return recall

if __name__ == "__main__":
    # Benchmark recall (noisy query cluster matches original): 0.9810 over 1000 samples
    codes, pages, vectors = load_all_vectors_and_ids("vectors_save")
    centroids, index, vectors = cluster_and_save(
        codes, pages, vectors,
        "clusters_output",
        n_clusters=2500,
        sample_size=500_000,
        batch_size=500_000,
    )
    benchmark_recall(centroids, index, vectors, n_queries=1000, noise_std=0.001)
