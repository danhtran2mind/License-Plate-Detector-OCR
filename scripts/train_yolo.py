from ultralytics import YOLO
from pathlib import Path
import sys
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Train a YOLOv10 model for license plate detection.')
parser.add_argument('--model-path', type=str, default='./ckpts/raw/yolo12n.pt', help='Path to the YOLO model weights')
parser.add_argument('--data', type=str, default='./datasets/yolo_standard_dataset/data.yaml', help='Path to the dataset YAML file')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch', type=int, default=64, help='Batch size for training')
parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--lrf', type=float, default=0.001, help='Final learning rate')
parser.add_argument('--device', type=str, default='0', help='Device to train on (e.g., 0, [0,1], or cpu)')
parser.add_argument('--project', type=str, default='./ckpts/finetune/runs', help='Directory to save training results')
parser.add_argument('--name', type=str, default='license_plate_detector', help='Name of the training run')
parser.add_argument('--save', action='store_true', default=True, help='Save training results')

# Parse arguments
args = parser.parse_args()

# Add parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, parent_dir)

# Load the YOLOv10 model
model = YOLO(args.model_path)

# Train the model
model.train(
    data=args.data,
    task='detect',
    mode='train',
    epochs=args.epochs,
    batch=args.batch,
    resume=args.resume,
    patience=args.patience,
    lr0=args.lr0,
    lrf=args.lrf,
    device=args.device,
    project=args.project,
    name=args.name,
    save=args.save
)