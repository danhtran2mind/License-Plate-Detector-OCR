# Training Arguments for YOLOv10 License Plate Detection

Below are the command-line arguments used to configure the training of a YOLOv10 model for license plate detection.

| Argument       | Type       | Default Value                              | Description                                                                 |
|----------------|------------|--------------------------------------------|-----------------------------------------------------------------------------|
| `--model`      | `str`      | `./ckpts/raw/yolo12n.pt`                   | Path to the model file or model name (e.g., "yolo12n.pt").                   |
| `--data`       | `str`      | `./datasets/yolo_standard_dataset/data.yaml`| Path to the dataset YAML file specifying the dataset configuration.          |
| `--epochs`     | `int`      | `100`                                      | Number of training epochs.                                                  |
| `--batch`      | `int`      | `64`                                       | Batch size for training.                                                    |
| `--resume`     | `boolean`  | `False`                                    | Resume training from the last checkpoint if set.                             |
| `--patience`   | `int`      | `20`                                       | Number of epochs to wait for improvement before early stopping.              |
| `--lr0`        | `float`    | `0.01`                                     | Initial learning rate for training.                                          |
| `--lrf`        | `float`    | `0.001`                                    | Final learning rate for training.                                           |
| `--device`     | `str`      | `0`                                        | Device to train on (e.g., `0` for GPU, `[0,1]` for multiple GPUs, or `cpu`). |
| `--project`    | `str`      | `./ckpts/finetune/runs`                    | Directory to save training results.                                          |
| `--name`       | `str`      | `license_plate_detector`                   | Name of the training run.                                                   |
| `--save`       | `boolean`  | `True`                                     | Save training results if set.                                                |