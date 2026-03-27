from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fal/AuraFace-v1",
    local_dir="models/auraface"
)
