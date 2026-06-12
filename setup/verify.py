"""端到端验证: prompt2image + image2text."""
import sys, os

def run(project_root: str) -> bool:
    print("\nVerifying...")
    sys.path.insert(0, os.path.join(project_root, "scripts"))
    from leuca.multimodal.api import MultimodalAPI

    api = MultimodalAPI()
    stats = api.stats
    assert stats["images"] > 0, "No images in index!"
    assert stats["texts"] > 0, "No texts in index!"

    r = api.prompt2image("test", topK=3)
    assert len(r["images"]) == 3, "prompt2image failed"
    r2 = api.image2text(f"data/images/img_0.jpg", topK=3)
    assert len(r2["texts"]) == 3, "image2text failed"

    print(f"[OK] Verified: {stats['images']} images, {stats['texts']} texts")
    print("[OK] All systems go!")
    return True
