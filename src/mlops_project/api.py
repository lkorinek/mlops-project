from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
import torch
from src.mlops_project.model import Model
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from google.cloud import storage

import anyio
import numpy as np
import pandas as pd
from evidently.metrics import DataDriftTable
from evidently.report import Report
from transformers import CLIPModel, CLIPProcessor

# Project root to get the models path
project_root = Path(__file__).resolve().parents[2]

MODEL_FILE_NAME = "trained_resnet50.ckpt" 
BUCKET_NAME = "your-gcp-bucket-name" # insert bucket name with trained model

async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI app by loading and cleaning up the model.
    """
    global model
    print("Loading model...")
    trained_model_path = project_root / "models" / MODEL_FILE_NAME

    # Starting code for gloud storage of trained model

    # if not trained_model_path.exists():
    #     print(f"Model file {MODEL_FILE_NAME} not found locally. Downloading from GCP...")
    #     download_model_from_gcp(trained_model_path)

    # Ensure the model file exists
    if trained_model_path.suffix == ".ckpt":
        # Trim the model name
        model_name = trained_model_path.stem.replace("trained_", "").split('-')[0]
        # Init model and load trained weights
        model = Model.load_from_checkpoint(trained_model_path, model_name=model_name, num_classes=1, map_location=torch.device("cpu"))
        model.eval()  
        print("Model loaded successfully!")
    else:
        raise RuntimeError("Model file not found or invalid")

    yield  # App continues running

    # Cleanup
    print("Cleaning up...")
    del model

def download_model_from_gcp(destination_path):
    """Download the model file from a GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE_NAME)
    blob.download_to_filename(destination_path)
    print(f"Model {MODEL_FILE_NAME} downloaded from GCP bucket {BUCKET_NAME}.")

app = FastAPI(lifespan=lifespan)

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Data drift stuff

# Load CLIP model and processor
print("Loading CLIP model and processor...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model and processor loaded.")

# Load datasets
train_images = torch.load(project_root / "data" / "processed" / "train_images.pt", weights_only=True)

# Function to preprocess images and get CLIP embeddings
def get_clip_embeddings(images, batch_size=32):
    """Process images and extract CLIP embeddings."""
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pil_images = []
        for img in batch:
            # Convert 1x1x224 (single channel) or similar shapes to [height, width, channels]
            img = img.squeeze()  # Remove singleton dimensions (e.g., 1x1x224 -> 224)
            if img.ndim == 2:  # Grayscale images
                img = np.stack([img] * 3, axis=-1)  # Convert to RGB by duplicating channels
            elif img.ndim == 3 and img.shape[0] in [1, 3]:  # Channels first
                img = img.transpose(1, 2, 0)  # Convert to height x width x channels
            
            # Convert to uint8 and PIL image
            img = (img * 255).clip(0, 255).astype("uint8")
            pil_images.append(Image.fromarray(img))
        
        # Process with CLIP
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        with torch.no_grad():
            batch_embeddings = clip_model.get_image_features(inputs["pixel_values"])
        embeddings.append(batch_embeddings.numpy())
    
    return np.vstack(embeddings)

# Initialize drift-related variables
train_embeddings = np.array([])  # This will be populated with train dataset embeddings

# Initialize your training embeddings
print("Getting training embeddings for drift detection..")
train_embeddings = get_clip_embeddings(train_images.numpy()) 
reference_df = pd.DataFrame(train_embeddings, columns=[f"Feature_{i}" for i in range(train_embeddings.shape[1])])
reference_df["Dataset"] = "X-RAY"
print("Reference dataset created.")
# Initialize current_df as a copy of reference_df (our starting data)
current_df = reference_df.copy()  # current_df that gets new data added.
current_df["Dataset"] = "Current"  


@app.get("/monitoring", response_class=HTMLResponse)
async def xray_monitoring():
    print("Currently computing monitoring file.")

    global current_df

    # Combine reference data with current data (new predictions)
    combined_df = pd.concat([reference_df, current_df], ignore_index=True)

    reference_data = combined_df[combined_df["Dataset"] == "X-RAY"].drop(columns=["Dataset"])
    current_data = combined_df[combined_df["Dataset"] == "Current"].drop(columns=["Dataset"])

    # Run Evidently Report for Data Drift
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("monitoring.html")

    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = await f.read() # Loads page (slowly)

    return HTMLResponse(content=html_content, status_code=200)


# Define class labels
true_labels = ["Normal", "Pneumonia"]

@app.post("/predict_pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):

    """
    Predict whether an uploaded chest X-ray image indicates pneumonia or is normal.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Open and process the uploaded file
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Transform the image for the classification model
    image_tensor = transform(image).unsqueeze(0)

    # Perform prediction for pneumonia classification
    with torch.no_grad():
        output = model(image_tensor)
        predicted = (torch.sigmoid(output) > 0.5).int()
        label = true_labels[predicted.item()]

    # Prepare the image for CLIP (no need to transform)
    inputs = processor(images=image_tensor, return_tensors="pt", padding=True, do_rescale=False)

    # Get embeddings from CLIP model
    with torch.no_grad():
        predicted_embedding = clip_model.get_image_features(inputs["pixel_values"])  # Correct method for CLIP model

    # Remove the extra dimensions (squeeze) and convert to numpy
    predicted_embedding = predicted_embedding.squeeze(0).numpy()  # (512,) embeddings

    # Convert the embeddings into a DataFrame
    new_embedding = pd.DataFrame([predicted_embedding], columns=[f"Feature_{i}" for i in range(predicted_embedding.shape[0])])  # predicted_embedding.shape[0] will give 512
    new_embedding["Dataset"] = "Current"

    # Add the new embedding to the current DataFrame
    global current_df 
    current_df = pd.concat([current_df, new_embedding], ignore_index=True)

    return JSONResponse(content={"label": label})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.mlops_project.api:app", host="0.0.0.0", port=8000, reload=True)

    # Disclaimer: I cannot get it to work by just running the file
    # I can however get it to run by running CLI: uvicorn src.mlops_project.api:app --host 0.0.0.0 --port 8000
    # And then test with an image like so: 
    # curl -X POST "http://localhost:8000/predict_pneumonia" -H "Content-Type: multipart/form-data" -F "file=@C:\person1_virus_6.JPEG"
    # In a terminal windows. I got the following output: {"label":"Normal"} 
    # When Navigating to http://localhost:8000/monitoring it computes the current datadrift.
    # Its quite slow since we're computing 512 embeddings comparisons, but yeah it should work.
    # When you predict on a new image it addes the embedding to the current dataset so the more images thats added the bigger drift will occur. 
    
    # NEEDS TO BE DONE:
    # Right way to handle models, but I did put some starting code to pull a trained model from a cloud bucket.
    # Automatic triggering if the drift in the Data Drift Summary report gets to a certain value.
    # I'm most likely missing something so add some stuff!