import numpy as np
import cv2
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(
    name="auraface",
    providers=["CPUExecutionProvider"],
    root="."
)

face_app.prepare(ctx_id=0, det_size=(640, 640))
print(face_app.models)

def get_face_embedding(image):
    faces = face_app.get(image)
    print("Faces detected:", len(faces))
    if not faces:
        return None

    faces.sort(
        key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
        reverse=True
    )
    
    return faces[0].normed_embedding.astype("float32")
