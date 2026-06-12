"""Chinese-CLIP 模型下载."""
def download():
    print("\nDownloading Chinese-CLIP base model (if not cached)...")
    try:
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        print("[OK] Chinese-CLIP model loaded")
        return True
    except Exception as e:
        print(f"[FAIL] Cannot download model: {e}")
        print("  Check network or set HF_ENDPOINT=https://hf-mirror.com")
        return False
