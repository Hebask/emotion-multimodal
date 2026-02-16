import argparse
import os
import subprocess
from pathlib import Path

def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle dataset into ./data")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. 'msambare/fer2013'")
    parser.add_argument("--out", default="data", help="Output directory (default: data)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run(["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out_dir)])

    for z in out_dir.glob("*.zip"):
        run(["python", "-c", f"import zipfile; z=zipfile.ZipFile(r'{z}'); z.extractall(r'{out_dir}'); z.close()"])
        print(f"Extracted: {z.name}")

    print("\nDone. Check:", out_dir.resolve())

if __name__ == "__main__":
    main()
