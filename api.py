from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
import uvicorn
import yaml
from model import create_model

app = FastAPI()

# Load model path and config path from environment variables
model_path = os.environ.get('MODEL_PATH', '/app/models/chess_classifier.pth')
config_path = os.environ.get('CONFIG_PATH', '/app/best_hyperparameters.yaml')

# Load the configuration
with open(config_path, 'r') as f:
    best_params = yaml.safe_load(f)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_info = torch.load(model_path, map_location=device, weights_only=True)

class_labels = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

confidence_threshold = 0.6

num_classes = len(class_labels)
num_units = best_params.get('num_units')
dropout_rate = best_params.get('dropout_rate')
model_name = best_params.get('model_name', 'resnet50')
model = create_model(num_classes=num_classes,
                     dropout_rate=dropout_rate,
                     model_name=model_name,
                     num_units=num_units)
try:
    model.load_state_dict(model_info['state_dict'])
except KeyError:
    model.load_state_dict(model_info)  

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class PredictionResult(BaseModel):
    class_label: str
    confidence: float

def predict_image(image: Image.Image):
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence = confidence.item()

            # Apply confidence threshold
            if confidence < confidence_threshold:
                return {
                    "class_label": "Uncertain",
                    "confidence": confidence
                }
            return {
                "class_label": class_labels[predicted_class.item()],
                "confidence": confidence
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResult)
async def classify_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload a JPEG or PNG image.")
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prediction = predict_image(image)
        return PredictionResult(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)