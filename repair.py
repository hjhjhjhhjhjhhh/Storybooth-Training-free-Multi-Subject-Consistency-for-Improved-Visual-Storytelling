import os
import json
from pathlib import Path
from huggingface_hub import snapshot_download

def patch_rope_scaling(model_id: str):
    """
    Auto-fix rope_scaling for Transformers<=4.36
    Convert Llama-3/Llama-3.1 rope_scaling to:
        {"type":"dynamic", "factor":X}
    """
    print(f"\n🔍 Locating model: {model_id}")

    # Download or locate local snapshot
    model_path = snapshot_download(model_id)
    config_path = Path(model_path) / "config.json"

    print(f"📄 config.json path: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError("config.json not found!")

    # Load config.json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Check if rope_scaling exists
    rope = cfg.get("rope_scaling", None)
    if rope is None:
        print("✔ No rope_scaling found. Nothing to patch.")
        return

    print("\n🧩 Before patch (original rope_scaling):")
    print(json.dumps(rope, indent=4))

    # Transformers<=4.36 only accepts:
    # {"type": ..., "factor": ...}
    factor = rope.get("factor", 1.0)

    cfg["rope_scaling"] = {
        "type": "dynamic",  # required
        "factor": factor    # preserve factor
    }

    # Save back
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    print("\n✅ After patch (transfomers-4.36 compatible rope_scaling):")
    print(json.dumps(cfg["rope_scaling"], indent=4))

    print("\n🎉 Patch complete! You can now load the model normally.\n")


if __name__ == "__main__":
    # Example:
    # patch_rope_scaling("meta-llama/Meta-Llama-3.1-8B-Instruct")

    MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    patch_rope_scaling(MODEL_ID)
