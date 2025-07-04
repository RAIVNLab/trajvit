# friedrichor/MSR-VTT


from huggingface_hub import snapshot_download
import os
import argparse

# Specify the repository and folder
    
def download_from_hf(repo_id, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, cache_dir=local_dir, repo_type="dataset")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    
    # Arguments for model configurations
    parser.add_argument("--repo_id", type=str, default='friedrichor/MSR-VTT', help="repo id from huggingface dataset")
    parser.add_argument("--local_dir", type=str, default='data', help="folder path to store the dataset")

    args = parser.parse_args()
    
    download_from_hf(args.repo_id, args.local_dir)