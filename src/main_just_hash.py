from scrap import batch_download_parallel_safe
from hash import embed_existing_galleries
from cluster_sep import cluster_all_batches
from merge import merge_cluster_batches
import gc
import math

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"✅ Deleted folder: {folder_path}")
    else:
        print(f"⚠️ Folder not found: {folder_path}")

import os
import shutil

def rename_merged_to_clusters(merged_dir="merged_clusters", target_dir="clusters"):
    # Remove existing "clusters" folder if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"🗑️ Deleted existing '{target_dir}' folder.")

    # Rename "merged_clusters" to "clusters"
    if os.path.exists(merged_dir):
        os.rename(merged_dir, target_dir)
        print(f"✅ Renamed '{merged_dir}' → '{target_dir}'")
    else:
        print(f"❌ '{merged_dir}' does not exist.")

def main_no_download():
    embed_existing_galleries()

    cluster_all_batches()

    merge_cluster_batches()

    delete_folder('vectors')
    delete_folder('clusters')
    rename_merged_to_clusters()

def count_webp_images(folder_path='galleries'):
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.webp'):
                image_count += 1
    return image_count

if __name__ == '__main__':
    max_id = 10_000
    download_batch = 128
    hash_cut_off = pow(2,14) # 16384
    cluster_offset = 0
    cur_batch = 0
    total_downloaded = 0

    print('Total number of batches',math.ceil(max_id/download_batch))
    while cur_batch*download_batch<=max_id:
        downloaded = 0
        while downloaded<=hash_cut_off:
            batch_download_parallel_safe(cur_batch*download_batch, (cur_batch+1)*download_batch,batch_id=str(cur_batch+1))
            cur_batch+=1
            downloaded = count_webp_images()
            total_downloaded+=total_downloaded
            print('Downloaded',downloaded,'images')
            if cur_batch*download_batch>max_id:
                print('Max_ID =',max_id,'reached terminating downloads.')
                break
            if downloaded<hash_cut_off:
                print('Downloading another batch since cut off is',hash_cut_off,'images.')
            else:
                print('Starting hashing due to exceeding cut off of',hash_cut_off,'images.')

        embed_existing_galleries()
        gc.collect()

        delete_folder('logging')
        delete_folder('galleries')
        gc.collect()


    print('Total images downloaded and hashed:',total_downloaded)