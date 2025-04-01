import os
import urllib.request
from pathlib import Path

def download(url, dest):
    """path"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[✓] Already downloaded: {dest}")
        return
    try:
        print(f"[↓] Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"[!] Failed to download {url}: {e}")

# === gnn-benchmark ===
gnn_benchmark_url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz"
amazon_datasets = {
    "amazon_electronics_computers": "data/amazon/amazon_electronics_computers.npz",
    "amazon_electronics_photo": "data/amazon/amazon_electronics_photo.npz"
}

coauthor_datasets = {
    "ms_academic_cs": "data/coauthor/ms_academic_cs.npz",
    "ms_academic_phy": "data/coauthor/ms_academic_phy.npz"
}

# === yandex heterophilous ===
yandex_base_url = "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data"
yandex_datasets = ["amazon-ratings", "minesweeper", "tolokers", "questions"]
yandex_files = ["edge.csv", "target.csv", "features.csv", "graph.json"]  

#  gnn-benchmark 
print("=== Downloading gnn-benchmark datasets ===")
for name, path in {**amazon_datasets, **coauthor_datasets}.items():
    url = f"{gnn_benchmark_url}/{name}.npz"
    download(url, path)

#  yandex 
print("=== Downloading yandex heterophilous datasets ===")
for dataset in yandex_datasets:
    for fname in yandex_files:
        url = f"{yandex_base_url}/{dataset}/{fname}"
        out_path = f"data/yandex/{dataset}/{fname}"
        download(url, out_path)

print("All datasets downloaded and saved.")
