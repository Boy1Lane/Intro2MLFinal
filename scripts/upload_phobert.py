"""Push a local fine-tuned PhoBERT folder to a Hugging Face Hub model repo.

Usage:
    HF_TOKEN=hf_xxx python scripts/upload_phobert.py output/phobert/phobert_best my-user/vihsd-phobert
"""
import os
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: upload_phobert.py <local_dir> <repo_id>", file=sys.stderr)
        return 2
    local_dir, repo_id = sys.argv[1], sys.argv[2]
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN env var (a HF write token).", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    print(f"Uploaded {local_dir} -> {repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
