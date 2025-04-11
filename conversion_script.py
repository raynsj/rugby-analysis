# conversion_script.py
import coremltools as ct
from ultralytics import YOLO

# Load original PyTorch model
model = YOLO('modelsv0.0.4/best(4).pt')

# Export to Core ML format
model.export(format='coreml')  # This creates 'best(4).mlmodel'
