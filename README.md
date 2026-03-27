# 🔍 Face Search API


## 📁 Project Structure

```
face-search-api/
│
├── api/
│   ├── main.py              # FastAPI entry point
│   ├── search_api.py        # Face search endpoint
│   └── index_api.py         # Index building endpoint
│
├── core/
│   ├── face_model.py        # InsightFace model loader
│   └── faiss_manager.py     # FAISS load/save/search logic
│
├── utils/
│   └── index_update.py      # ZIP → embeddings → index update
│
├── config.py                # Config variables (paths, thresholds)
├── download_model.py        # Download AuraFace model
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Use Linux or WSL otherwise insightface won't run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/face-search-api.git
cd face-search-api
```

### 2. Create Environment

```bash
conda create -n face_search python=3.10 -y
conda activate face_search
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the AuraFace Model

> ⚠️ **Required before running the API.**

```bash
python download_model.py
```

This creates the `models/auraface/` directory with the required weights.

### 5. Prepare Data Directory

The API expects the following structure:

```
data/
├── gallery_index.faiss
├── gallery_meta.json
└── Gallery/
```

> ⚠️ These files are **not included** in the repo. Generate them using the `/build-index` endpoint (see below).

---

## 🚀 Run the API

```bash
uvicorn api.main:app --reload
```

Then open the interactive Swagger docs at:

```
http://127.0.0.1:8000/docs
```

---

## 🌐 API Endpoints

### `POST /search` — Search by Face

Upload a `.jpg` or `.png` image to find matching faces in the indexed gallery.

**Response:**
```json
{
  "matches": [
    {
      "image_path": "Gallery/person_001.jpg",
      "score": 0.87
    }
  ],
  "count": 10
}
```

---

### `POST /build-index` — Bulk Index from ZIP

Upload a `.zip` file containing images. The API will:

1. Extract all images
2. Detect faces
3. Generate embeddings
4. Remove near-duplicates
5. Update the FAISS index + metadata

**Response:**
```json
{
  "added": 120,
  "duplicates_skipped": 30
}
```

---

## 🧠 How It Works

### Face Embedding
Uses **InsightFace (AuraFace)** to generate 512-dimensional, L2-normalized face vectors from detected faces in an image.

### Similarity Search
FAISS `IndexFlatIP` performs inner product search over normalized vectors, which is equivalent to **cosine similarity** — giving fast and accurate nearest-neighbor lookup.

### Duplicate Detection
Before indexing, each new face embedding is compared against existing entries. Faces with similarity above the deduplication threshold are skipped:

```python
SIM_THRESHOLD_DUP = 0.9
```

---

## ⚠️ Important Notes

- Run in a **WSL or Linux** environment (Windows native may have issues)
- Ensure `models/` and `data/` directories exist before starting
- Only `.jpg` and `.png` image formats are supported
- The FAISS index and metadata JSON must stay in sync — do not manually edit them

---

## 🛠️ Common Issues

| Error | Cause | Fix |
|---|---|---|
| `No face detected` | Image is unclear or non-frontal | Use a clear, frontal face photo |
| `NoneType has no attribute shape` | Invalid/corrupt image upload | Check the uploaded file |
| `faiss not defined` | Missing import | Add `import faiss` at the top |

---

## 📈 Roadmap

- [ ] Face clustering (identity grouping)
- [ ] GPU acceleration support
- [ ] Vector DB integration (Milvus / Pinecone)
- [ ] Async indexing pipeline
- [ ] Docker deployment

---

## 👨‍💻 Author

**Shashank Mishra**  
B.Tech — Data Science & AI | IIIT Naya Raipur

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub — it helps a lot!
