import struct
import gzip
import os
import re

def write_string(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<H', len(b)))
    f.write(b)

def write_string_list(f, lst):
    f.write(struct.pack('<B', len(lst)))
    for s in lst:
        write_string(f, s)

def write_metadata_entry(f, entry):
    f.write(struct.pack('<i', int(entry['code'])))
    write_string(f, entry.get('name', ''))
    write_string_list(f, entry.get('top_tags', []))
    write_string_list(f, entry.get('top_chars', []))
    thumb_bytes = entry.get('thumbnail', b'')
    f.write(struct.pack('<I', len(thumb_bytes)))
    f.write(thumb_bytes)

def read_string(f):
    len_bytes = f.read(2)
    if not len_bytes or len(len_bytes) < 2:
        return '', 0
    l = struct.unpack('<H', len_bytes)[0]
    s = f.read(l).decode('utf-8')
    return s, 2 + l

def read_string_list(f):
    count_bytes = f.read(1)
    if not count_bytes or len(count_bytes) < 1:
        return [], 1
    count = struct.unpack('<B', count_bytes)[0]
    lst = []
    total = 1
    for _ in range(count):
        s, l = read_string(f)
        lst.append(s)
        total += l
    return lst, total

def read_metadata_entry(f):
    code_bytes = f.read(4)
    if not code_bytes or len(code_bytes) < 4:
        return None, 0
    code = struct.unpack('<i', code_bytes)[0]
    name, l1 = read_string(f)
    top_tags, l2 = read_string_list(f)
    top_chars, l3 = read_string_list(f)
    thumb_len_bytes = f.read(4)
    if not thumb_len_bytes or len(thumb_len_bytes) < 4:
        return None, 0
    thumb_len = struct.unpack('<I', thumb_len_bytes)[0]
    thumb = f.read(thumb_len)
    total = 4 + l1 + l2 + l3 + 4 + thumb_len
    return {'code': code, 'name': name, 'top_tags': top_tags, 'top_chars': top_chars, 'thumbnail': thumb}, total

def combine_metadata_from_folder():
    folder_path = "clusters/clusters"  # <== Change this to your folder path
    output_folder = "meta"             # output folder name
    batch_size = 500                  # batch size to match JS logic

    pattern = re.compile(r'cluster_\d+_meta\.bin\.gz')
    files = [f for f in os.listdir(folder_path) if pattern.match(f)]
    files.sort()  # sort filenames

    files = [os.path.join(folder_path, f) for f in files]
    print(f"Found {len(files)} files matching pattern in {folder_path}")

    unique_entries = {}

    for filename in files:
        print(f"Reading {filename} ...")
        with gzip.open(filename, 'rb') as f:
            while True:
                entry, size = read_metadata_entry(f)
                if entry is None or size == 0:
                    break
                code = entry['code']
                if code not in unique_entries:
                    unique_entries[code] = entry

    total_entries = len(unique_entries)
    print(f"Total unique entries found: {total_entries}")

    # Group entries by batch index = code // batch_size
    batches = {}
    max_batch_index = 0
    for code, entry in unique_entries.items():
        batch_index = code // batch_size
        if batch_index > max_batch_index:
            max_batch_index = batch_index
        batches.setdefault(batch_index, []).append(entry)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Writing {max_batch_index+1} batches into folder '{output_folder}'...")

    for batch_i in range(max_batch_index + 1):
        batch_entries = batches.get(batch_i, [])
        output_file = os.path.join(output_folder, f"meta_{batch_i}.bin.gz")
        print(f" Writing batch {batch_i} with {len(batch_entries)} entries -> {output_file}")
        with gzip.open(output_file, 'wb') as f_out:
            for entry in batch_entries:
                write_metadata_entry(f_out, entry)

    print("Done.")

if __name__ == "__main__":
    combine_metadata_from_folder()
