import wget
import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_files

def download_model_ckpts(args):
    """
    Download the yolo12n.pt model from a URL and the best.pt (or all models if specified) from a Hugging Face repository.
    """
    # Download yolo12n.pt from GitHub URL
    model_url = args.url
    output_dir = args.output_dir
    raw_output_dir = os.path.join(output_dir, 'raw')  # Subdirectory for wget download
    
    # Extract filename from URL
    filename = model_url.split("/")[-1]
    output_path = os.path.join(raw_output_dir, filename)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download with wget
    wget.download(
        model_url,
        out=output_path,
        bar=wget.bar_adaptive  # Show progress bar
    )
    print(f"\nDownloaded model from {model_url} to {output_path}")

    # Download from Hugging Face repository
    hf_repo = args.hf_repo
    if args.best_models_only:
        # Download only best.pt
        model_file = "yolo/finetune/runs/license_plate_detector/weights/best.pt"
        hf_output_path = os.path.join(output_dir, "best.pt")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(hf_output_path), exist_ok=True)
        
        # Download the specific file
        downloaded_path = hf_hub_download(
            repo_id=hf_repo,
            filename=model_file,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"\nDownloaded model file from {hf_repo} to {downloaded_path}")
    else:
        # Download all files from the repository
        repo_files = list_repo_files(repo_id=hf_repo)
        for model_file in repo_files:
            # Skip non-model files if needed or filter by specific directory
            if model_file.startswith("yolo/finetune/runs/license_plate_detector/weights/"):
                hf_output_path = os.path.join(output_dir, os.path.basename(model_file))
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(hf_output_path), exist_ok=True)
                
                # Download each file
                downloaded_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=model_file,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False
                )
                print(f"\nDownloaded model file {model_file} from {hf_repo} to {downloaded_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download yolo12n.pt from URL and best.pt (or all models) from Hugging Face repository.")
    parser.add_argument('--url', type=str,
                        default='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt',
                        help='URL of the yolo12n.pt model to download via wget')
    parser.add_argument('--output-dir', type=str, default='./ckpts',
                        help='Base output directory for downloaded model files')
    parser.add_argument('--hf-repo', type=str, default='danhtran2mind/license-plate-detector-ocr',
                        help='Hugging Face repository ID to download model file from')
    parser.add_argument('--best-models-only', action='store_true',
                        help='If set, download only best.pt; otherwise, download all model files from Hugging Face repository')

    # Parse arguments
    args = parser.parse_args()

    download_model_ckpts(args)
