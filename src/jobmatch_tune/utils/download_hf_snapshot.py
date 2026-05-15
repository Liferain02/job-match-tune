from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests


MODEL_FILES = {
    "Qwen/Qwen3-14B": [
        ".gitattributes",
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model.safetensors.index.json",
        "model-00001-of-00008.safetensors",
        "model-00002-of-00008.safetensors",
        "model-00003-of-00008.safetensors",
        "model-00004-of-00008.safetensors",
        "model-00005-of-00008.safetensors",
        "model-00006-of-00008.safetensors",
        "model-00007-of-00008.safetensors",
        "model-00008-of-00008.safetensors",
    ],
    "Qwen/Qwen3-1.7B": [
        ".gitattributes",
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ],
}


def model_base_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main"


def download_file(session: requests.Session, repo_id: str, filename: str, local_dir: Path, retries: int) -> None:
    url = f"{model_base_url(repo_id)}/{filename}"
    destination = local_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    attempt = 0
    while True:
        local_size = destination.stat().st_size if destination.exists() else 0
        headers = {"Range": f"bytes={local_size}-"} if local_size else {}
        mode = "ab" if local_size else "wb"
        try:
            with session.get(url, headers=headers, stream=True, timeout=(15, 120), allow_redirects=True) as response:
                if response.status_code == 416:
                    print(f"already complete: {destination}")
                    return
                response.raise_for_status()
                total = response.headers.get("Content-Length")
                total_display = f"{int(total) / (1024**3):.2f} GiB" if total and total.isdigit() else "unknown"
                print(f"downloading {filename} from offset {local_size} ({total_display})")
                with destination.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print(f"done {filename}")
            return
        except requests.RequestException as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"failed to download {filename}: {exc}") from exc
            sleep_seconds = min(30, 2 * attempt)
            print(f"retry {attempt}/{retries} for {filename}: {exc}")
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()

    files = MODEL_FILES.get(args.repo_id)
    if not files:
        raise ValueError(f"unsupported repo id: {args.repo_id}")

    local_dir = Path(args.local_dir)
    session = requests.Session()
    session.headers.update({"User-Agent": "job-match-tune-downloader/1.0"})
    for filename in files:
        download_file(session, args.repo_id, filename, local_dir, args.retries)


if __name__ == "__main__":
    main()
