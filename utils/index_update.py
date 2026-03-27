import zipfile
import os
import json
import cv2
import numpy as np
import faiss
import pathlib
from insightface.app import FaceAnalysis

# 🔥 Load face model ONCE
face_app = FaceAnalysis(name="auraface", providers=["CPUExecutionProvider"], root=".")
face_app.prepare(ctx_id=0, det_size=(640, 640))

INDEX_PATH = "data/gallery_index.faiss"
META_PATH  = "data/gallery_meta.json"
GALLERY_DIR = "data/Gallery"

SIM_THRESHOLD_DUP = 0.9


def update_index_from_zip(file):
    zip_path = file.name

    extract_dir = "temp_upload"
    os.makedirs(extract_dir, exist_ok=True)

    print("📦 Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 🔥 Load existing index + metadata
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    new_embeddings = []
    new_metadata = []

    added = 0
    skipped_dup = 0
    skipped_no_face = 0

    for root, _, files in os.walk(extract_dir):
        for file in files:

            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # 🔥 MULTI FACE (same as your Colab)
            faces = face_app.get(img)

            if not faces:
                skipped_no_face += 1
                continue

            for face_idx, face in enumerate(faces):
                emb = face.normed_embedding.astype(np.float32)

                emb_2d = emb.reshape(1, -1)
                faiss.normalize_L2(emb_2d)

                # 🔥 DUPLICATE CHECK
                if index.ntotal > 0:
                    score, _ = index.search(emb_2d, 1)

                    if score[0][0] > SIM_THRESHOLD_DUP:
                        skipped_dup += 1
                        continue

                # 🔥 MOVE IMAGE TO GALLERY
                os.makedirs(GALLERY_DIR, exist_ok=True)
                new_path = os.path.join(GALLERY_DIR, file)

                if not os.path.exists(new_path):
                    os.rename(img_path, new_path)

                new_embeddings.append(emb)

                new_metadata.append({
                    "image_path": os.path.abspath(new_path),
                    "face_idx": face_idx,
                    "bbox": face.bbox.astype(int).tolist(),
                    "subfolder": pathlib.Path(new_path).parent.name
                })

                added += 1

    # 🔥 ADD TO FAISS
    if new_embeddings:
        emb_matrix = np.stack(new_embeddings).astype(np.float32)
        faiss.normalize_L2(emb_matrix)

        index.add(emb_matrix)

        metadata.extend(new_metadata)

        faiss.write_index(index, INDEX_PATH)

        with open(META_PATH, "w") as f:
            json.dump(metadata, f)

    print(f"""
    ✅ Added: {added}
    ⚠️ Duplicates skipped: {skipped_dup}
    ⚠️ No face: {skipped_no_face}
    """)

    return f"Added: {added}, Duplicates skipped: {skipped_dup}"