"""OpenVINO GPU 模型导出 (Intel Arc / iGPU)."""
import os

def export(project_root: str) -> bool:
    print("\nExporting OpenVINO IR models...")
    try:
        import torch, torch.nn as nn
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        import openvino as ov

        model = ChineseCLIPModel.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True).eval()
        processor = ChineseCLIPProcessor.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True)

        ov_dir = os.path.join(project_root, "data", "openvino")
        os.makedirs(ov_dir, exist_ok=True)

        # text encoder
        text_path = os.path.join(ov_dir, "text_encoder.xml")
        if not os.path.exists(text_path):
            class TE(nn.Module):
                def __init__(self, m): super().__init__()
                def forward(self, ids, mask):
                    out = m.text_model(input_ids=ids, attention_mask=mask)
                    return m.text_projection(out.last_hidden_state[:, 0, :])
            te = TE(model).eval()
            ti = processor(text=["test"], return_tensors="pt", padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                traced = torch.jit.trace(te, (ti["input_ids"], ti["attention_mask"]))
            ov.save_model(ov.convert_model(traced, example_input=(ti["input_ids"], ti["attention_mask"])), text_path)
            print("  [OK] text_encoder")

        # vision encoder
        vision_path = os.path.join(ov_dir, "vision_encoder.xml")
        if not os.path.exists(vision_path):
            class VE(nn.Module):
                def __init__(self, m): super().__init__()
                def forward(self, px):
                    out = m.vision_model(pixel_values=px)
                    return m.visual_projection(out.last_hidden_state[:, 0, :])
            ve = VE(model).eval()
            with torch.no_grad():
                traced = torch.jit.trace(ve, torch.randn(1, 3, 224, 224))
            ov.save_model(ov.convert_model(traced, example_input=torch.randn(1, 3, 224, 224)), vision_path)
            print("  [OK] vision_encoder")

        print("[OK] OpenVINO models exported (GPU inference ready)")
        return True
    except Exception as e:
        print(f"[WARN] OpenVINO export failed: {e}")
        return False
