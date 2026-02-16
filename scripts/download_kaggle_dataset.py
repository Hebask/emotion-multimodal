import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def download_one(slug: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"])

    meta = {
        "slug": slug,
        "out_dir": str(out_dir),
        "downloaded_at": datetime.utcnow().isoformat() + "Z"
    }
    (out_dir / "_kaggle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✅ Downloaded: {slug} -> {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets into ./data (Kaggle-first)")
    parser.add_argument("--config", default="scripts/kaggle_datasets.json", help="JSON config file path")
    parser.add_argument("--only", default="", help="Comma-separated: text,image,audio,video (optional)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    only = [x.strip() for x in args.only.split(",") if x.strip()]
    keys = only if only else list(cfg.keys())

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)

    for k in keys:
        if k not in cfg:
            raise ValueError(f"Unknown key '{k}'. Available: {list(cfg.keys())}")
        slug = cfg[k]["slug"]
        out = Path(cfg[k]["out"])
        download_one(slug, out)

    print("\nDone. Your Kaggle datasets are in ./data/raw/* (and ignored by git).")

if __name__ == "__main__":
    main()
