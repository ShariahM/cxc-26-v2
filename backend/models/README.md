# YOLOv8 Models Directory

## Place Your Trained YOLOv8 Model Here

This directory should contain your trained YOLOv8 model file.

### Expected Model File

- **Filename**: `best.pt` (or update the path in `app/models/detection.py`)
- **Format**: PyTorch `.pt` format
- **Type**: YOLOv8 model trained on NFL player detection

### Model Classes

Your model should be trained to detect the following classes (or you can customize in `detection.py`):

- `player` (id: 0)
- `quarterback` (id: 1)
- `receiver` (id: 2)
- `defender` (id: 3)
- `ball` (id: 4)

### If You Don't Have a Model Yet

The application will fall back to using a default YOLOv8n model for development/testing purposes. However, for NFL-specific analysis, you should train your own model.

### Training Your Own Model

1. Collect and label NFL gameplay videos with player positions
2. Use Ultralytics YOLOv8 to train:
   ```python
   from ultralytics import YOLO
   
   model = YOLO('yolov8n.pt')
   results = model.train(data='your_dataset.yaml', epochs=100)
   ```
3. Save the best model as `best.pt` and place it here

### Updating Model Path

If your model has a different name, update line 11 in `backend/app/models/detection.py`:

```python
def __init__(self, model_path: str = "models/your_model_name.pt"):
```
