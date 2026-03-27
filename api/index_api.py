from fastapi import APIRouter, UploadFile, File
import zipfile
import os
import cv2
import numpy as np
import pathlib
import faissdse42

from core.face_model import face_app
from core.faiss_manager import load_index, save_index

router = APIRouter()

SIM_THRESHOLD_DUP = 0.9
GALLERY_DIR = "data/Gallery"


@router.post("/build-index")
async def build_index(file: UploadFile = File(...)):

    zip_path = "temp.zip"
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    extract_dir = "temp_extract"
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    index, metadata = load_index()

    new_embeddings = []
    new_metadata = []

    added = 0
    skipped_dup = 0

    for root, _, files in os.walk(extract_dir):
        for file in files:

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            faces = face_app.get(img)

            if not faces:
                continue

            for face_idx, face in enumerate(faces):
                emb = face.normed_embedding.astype(np.float32)
                emb_2d = emb.reshape(1, -1)
                faiss.normalize_L2(emb_2d)

                if index.ntotal > 0:
                    score, _ = index.search(emb_2d, 1)
                    if score[0][0] > SIM_THRESHOLD_DUP:
                        skipped_dup += 1
                        continue

                new_path = os.path.join(GALLERY_DIR, file)
                os.makedirs(GALLERY_DIR, exist_ok=True)

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

    if new_embeddings:
        emb_matrix = np.stack(new_embeddings).astype(np.float32)
        faiss.normalize_L2(emb_matrix)

        index.add(emb_matrix)
        metadata.extend(new_metadata)

        save_index(index, metadata)

    return {
        "added": added,
        "duplicates_skipped": skipped_dup
    }