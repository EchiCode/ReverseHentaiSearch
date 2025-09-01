import os
import orjson
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import shutil
import gzip
import struct


def embed_existing_galleries(
    gallery_root="galleries",
    output_dir="vectors",
    batch_size=128,  # be careful the batching is per gallery, so for say 1024 it will work for most before crashing at ~50k...
    save_every=1_000_000_000,
    resize=(256, 256),
    model_name="facebook/dino-vits16",
    offset = 0,
    device=None,
    delete_gallery=True
):
    os.makedirs(output_dir, exist_ok=True)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    all_gallery_folders = sorted(
        [f for f in os.listdir(gallery_root) if os.path.isdir(os.path.join(gallery_root, f)) and f.isdigit()],
        key=lambda x: int(x)
    )

    # Automatically determine batch number from existing files
    existing_batches = [
        f for f in os.listdir(output_dir) if f.startswith("dino_vectors_batch_") and f.endswith(".bin")
    ]
    existing_ids = [
        int(f.replace("dino_vectors_batch_", "").replace(".bin", ""))
        for f in existing_batches if f.replace("dino_vectors_batch_", "").replace(".bin", "").isdigit()
    ]
    batch_num = max(existing_ids, default=-1) + 1+offset
    batch_orig = batch_num

    db = []
    processed = 0

    for gallery_id in tqdm(all_gallery_folders, desc="Embedding galleries"):
        gallery_path = os.path.join(gallery_root, gallery_id)
        image_files = sorted([
            os.path.join(gallery_path, f)
            for f in os.listdir(gallery_path)
            if f.lower().endswith((".webp",".png", ".jpg", ".jpeg"))
        ])

        if not image_files:
            continue

        for batch_idx in range(0, len(image_files), batch_size):
            batch_files = image_files[batch_idx:batch_idx + batch_size]
            imgs = []
            meta = []

            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(resize, Image.LANCZOS)
                    page = os.path.splitext(os.path.basename(img_path))[0]
                    imgs.append(img)
                    meta.append((gallery_id, page))
                except Exception:
                    continue

            if not imgs:
                continue

            inputs = processor(images=imgs, return_tensors="pt").to(device)

            with torch.no_grad():
                features = model(**inputs).last_hidden_state[:, 0]  # Use CLS token
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                features = features.cpu().numpy()

            for (code, page), vec in zip(meta, features):
                db.append({"code": code, "page": page, "vector": vec.tolist()})
                processed += 1

            if len(db) >= save_every:
                out_path = os.path.join(output_dir, f"dino_vectors_batch_{batch_num}.bin")
                with open(out_path, "wb") as f:
                    for entry in db:
                        code = int(entry['code'])
                        page = int(entry['page'])
                        vector = np.array(entry['vector'], dtype=np.float16)
                        f.write(struct.pack('<i', code))
                        f.write(struct.pack('<h', page))
                        f.write(vector.tobytes())
                print(f"\u2705 Saved batch {batch_num} with {len(db)} vectors \u2192 {out_path}")
                db = []
                batch_num += 1
        if delete_gallery:
            shutil.rmtree(gallery_path)

    if db:
        out_path = os.path.join(output_dir, f"dino_vectors_batch_{batch_num}.bin")
        with open(out_path, "wb") as f:
            for entry in db:
                code = int(entry['code'])
                page = int(entry['page'])
                vector = np.array(entry['vector'], dtype=np.float16)
                f.write(struct.pack('<i', code))
                f.write(struct.pack('<h', page))
                f.write(vector.tobytes())
        print(f"\u2705 Saved final batch {batch_num} with {len(db)} vectors \u2192 {out_path}")
        batch_num += 1

    print(f"🎉 Done! Embedded {processed} images across {batch_num-batch_orig} total batches.")
    return batch_num