from ultralytics import YOLO

_model_cache = None

def load_yolo_model(model_path):
    """Load and cache the YOLO model from the specified path."""
    global _model_cache
    if _model_cache is None:
        try:
            _model_cache = YOLO(model_path, verbose=False)
        except Exception as e:
            raise Exception(f"Error loading YOLO model: {e}")
    return _model_cache

def yolo_infer(model_path, input_data):
    """Perform YOLO inference on input data using the cached model."""
    try:
        model = load_yolo_model(model_path)
        results = model(input_data, verbose=False)
        return results
    except Exception as e:
        print(f"Error during YOLO inference: {e}")
        return []

if __name__ == "__main__":
    print("This module is intended for import, not direct execution.")