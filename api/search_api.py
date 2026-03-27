from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
from core.face_model import get_face_embedding
from core.faiss_manager import load_index, search

router = APIRouter()

index, metadata = load_index()


@router.post("/search")
async def search_face_api(file: UploadFile = File(...)):

    contents = await file.read()

    if contents is None or len(contents) == 0:
        return {"error": "Empty file uploaded"}

    # 🔥 Convert bytes → numpy array
    np_arr = np.frombuffer(contents, np.uint8)

    # 🔥 Decode image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 🚨 CRITICAL CHECK
    if img is None:
        return {"error": "Invalid image file or unsupported format"}

    print("DEBUG → Image shape:", img.shape)

    emb = get_face_embedding(img)

    if emb is None:
        return {"error": "No face detected"}

    results = search(index, metadata, emb)

    return {
        "matches": results,
        "count": len(results)
    }