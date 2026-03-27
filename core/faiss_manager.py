import faiss
import json
import numpy as np

INDEX_PATH = "data/gallery_index.faiss"
META_PATH  = "data/gallery_meta.json"


def load_index():
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    return index, metadata


def save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)


def search(index, metadata, emb, top_k=10000, threshold=0.50):
    emb = emb.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(emb)

    scores, indices = index.search(emb, top_k)
    scores, indices = scores[0], indices[0]

    results = []

    for score, idx in zip(scores, indices):
        if idx == -1 or score < threshold:
            continue

        results.append({
            "image_path": metadata[idx]["image_path"],
            "score": float(score)
        })

    return results