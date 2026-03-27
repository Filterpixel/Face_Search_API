from PIL import Image

def load_images(results):
    images = []
    captions = []

    for path, score in results:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            captions.append(f"{path.split('/')[-1]} | {score:.3f}")
        except:
            continue

    return images, captions