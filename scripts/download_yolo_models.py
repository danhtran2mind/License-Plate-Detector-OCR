import wget
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Download model files using wget.")
parser.add_argument('--url', type=str,
                    default='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt',
                    help='URL of the model to download')
parser.add_argument('--output-dir', type=str, default='./ckpts/raw',
                    help='Output directory for downloaded models')

# Parse arguments
args = parser.parse_args()

# Get URL and output directory from arguments
model_urls = [args.url]  # Single URL as a list for consistency
output_dir = args.output_dir

for url in model_urls:
    # Extract filename from URL
    filename = url.split("/")[-1]
    output_path = os.path.join(output_dir, filename)  # Construct full path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download with additional wget arguments
    wget.download(
        url,
        out=output_path,
        bar=wget.bar_adaptive,  # Show progress bar
    )
    print(f"\nDownloaded model to {output_path}")