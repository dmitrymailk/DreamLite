# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Утилита для скачивания весов DreamLite с Hugging Face Hub.

Пример:
    python download_model.py
    python download_model.py --repo_id carlofkl/DreamLite-base --local_dir /code/models/DreamLite-base
"""

import argparse

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download DreamLite weights from HuggingFace")
    parser.add_argument("--repo_id", type=str, default="carlofkl/DreamLite-mobile")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--local_dir", type=str, default="/code/models/DreamLite-mobile")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Downloading {args.repo_id} (revision={args.revision}) -> {args.local_dir}")
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
    )
    print(f"Done. Weights are at: {path}")


if __name__ == "__main__":
    main()
