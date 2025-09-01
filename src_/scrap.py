import os
import orjson
import cloudscraper
from PIL import Image, ImageFile
from io import BytesIO
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import base64

ImageFile.LOAD_TRUNCATED_IMAGES = True
scraper = cloudscraper.create_scraper()

OUTPUT_DIR = "galleries"
METADATA_PATH = "metadata.json"  # Saved in main folder
IMAGE_SIZE = (256, 256)


def download_image(media_id, idx, page_info, size=IMAGE_SIZE, gallery_folder=None):
    ext = {"j": "jpg", "p": "png", "g": "gif"}.get(page_info["t"], "jpg")
    img_url = f"https://i.nhentai.net/galleries/{media_id}/{idx + 1}.{ext}"
    try:
        img_resp = scraper.get(img_url, timeout=15)
        img_resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download image {img_url}: {e}")

    content_type = img_resp.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise RuntimeError(f"URL {img_url} did not return an image (Content-Type: {content_type})")

    try:
        img = Image.open(BytesIO(img_resp.content)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image from {img_url}: {e}")

    img = img.resize(size, Image.LANCZOS)

    if gallery_folder:
        os.makedirs(gallery_folder, exist_ok=True)
        out_path = os.path.join(gallery_folder, f"{idx + 1:03d}.webp")
        img.save(out_path, format='WebP')

    return img


def download_and_extract_metadata(gallery_id, output_dir=OUTPUT_DIR, size=IMAGE_SIZE, max_img_workers=4):
    try:
        api_url = f"https://nhentai.net/api/gallery/{gallery_id}"
        response = scraper.get(api_url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        return None, f"❌ Failed to fetch gallery metadata {gallery_id}: {e}"

    try:
        gallery_data = response.json()
    except Exception as e:
        return None, f"❌ Failed to parse JSON for gallery {gallery_id}: {e}"

    media_id = gallery_data.get("media_id")
    if media_id is None:
        return None, f"❌ Gallery {gallery_id} missing media_id"

    image_info = gallery_data.get("images", {}).get("pages", [])
    if not image_info:
        return None, f"❌ Gallery {gallery_id} has no images info"

    full_output = os.path.join(output_dir, str(gallery_id))
    os.makedirs(full_output, exist_ok=True)

    imgs = [None] * len(image_info)
    with ThreadPoolExecutor(max_workers=max_img_workers) as img_executor:
        futures = {
            img_executor.submit(download_image, media_id, i, page, size, full_output): i
            for i, page in enumerate(image_info)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                imgs[idx] = future.result()
            except Exception as e:
                return None, f"❌ Gallery {gallery_id} image {idx+1} download failed: {e}"

    # Create binary thumbnail from first image
    tumb = None
    if imgs[0]:
        thumb_img = imgs[0]# .resize((196, 196), Image.LANCZOS)
        buffer = BytesIO()
        thumb_img.save(buffer, format='WebP', quality=80, lossless=False)
        buffer.seek(0)
        tumb_b = buffer.read()  # Store as binary data
        tumb = base64.b64encode(tumb_b).decode('utf-8')

    # Titles: prefer pretty, then English, then Japanese
    title = (
        gallery_data.get("title", {}).get("pretty")
        or gallery_data.get("title", {}).get("english")
        or gallery_data.get("title", {}).get("japanese")
        or "Untitled"
    )

    # Extract tags and characters with counts
    tags = gallery_data.get("tags", [])
    all_tags = [(tag["name"], tag.get('count', 0)) for tag in tags if tag.get("type") == "tag"]
    all_chars = [(tag["name"], tag.get('count', 0)) for tag in tags if tag.get("type") == "character"]
    top_tags = [t[0] for t in sorted(all_tags, key=lambda x: -x[1])[:5]]
    top_chars = [t[0] for t in sorted(all_chars, key=lambda x: -x[1])[:5]]

    meta = {
        "id": gallery_id,
        "name": title.strip(),
        "top_tags": top_tags,
        "top_chars": top_chars,
        "thumbnail": tumb,  # Store path to binary thumbnail file
    }

    return meta, f"✅ Gallery {gallery_id} downloaded"


def batch_download_parallel_safe(start=0, end=1000, max_workers=12, batch_id="default"):
    log_dir = "logging"
    os.makedirs(log_dir, exist_ok=True)

    downloaded_file = os.path.join(log_dir, "downloaded.txt")

    gallery_ids = list(range(start, end))
    downloaded_set = set()

    if os.path.exists(downloaded_file):
        with open(downloaded_file, "r") as f:
            downloaded_set = set(int(line.strip()) for line in f if line.strip().isdigit())

    to_download = [gid for gid in gallery_ids if gid not in downloaded_set]
    print(f"⏳ {len(to_download)} galleries to download")

    metadata = []
    messages = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
        total=len(to_download), desc=f"Batch {batch_id} Downloading galleries"
    ) as pbar:
        futures = {executor.submit(download_and_extract_metadata, gid, OUTPUT_DIR, IMAGE_SIZE): gid for gid in to_download}
        with open(downloaded_file, "a") as f:
            for future in as_completed(futures):
                gid = futures[future]
                meta, result = future.result()
                # messages.append(result)
                if meta:
                    metadata.append(meta)
                    f.write(f"{gid}\n")
                    f.flush()
                pbar.update(1)

    # print("\n".join(messages))

    existing_metadata = []
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            try:
                existing_metadata = orjson.loads(f.read())
            except orjson.JSONDecodeError:
                existing_metadata = []

    existing_metadata.extend(metadata)
    with open(METADATA_PATH, "wb") as f:
        f.write(orjson.dumps(existing_metadata))

    print(f"✅ Appended {len(metadata)} new entries to {METADATA_PATH}")


if __name__ == "__main__":
    # Example usage
    batch_download_parallel_safe(0, 10, max_workers=6, batch_id="batch1")
