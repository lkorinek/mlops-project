from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
import torch
from src.mlops_project.model import Model
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from google.cloud import storage
from pydantic import BaseModel
import pickle

import anyio
import numpy as np
import pandas as pd
from evidently.metrics import DataDriftTable
from evidently.report import Report
from transformers import CLIPModel, CLIPProcessor

# Project root to get the models path
project_root = Path(__file__).resolve().parents[2]

BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "trained_models_mlops")  # Update with your bucket name
LOCAL_MODEL_DIR = project_root / "cloud_storage_models"

# Initialize global variables for model, CLIP model, and processor
model = None
clip_model = None
processor = None
current_model_name = None

# Image transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# Initialize CLIP model and processor
def initialize_clip():
    global clip_model, processor
    if clip_model is None or processor is None:
        print("Loading CLIP model and processor...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model and processor loaded.")


# List available models in the GCP bucket
def list_models_in_gcp():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return [blob.name for blob in bucket.list_blobs() if blob.name.endswith(".ckpt")]


# Download a model from GCP
def download_model_from_gcp(model_name):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Remove 'trained_models/' prefix if it's included in the model name
    if model_name.startswith("trained_models/"):
        model_name = model_name[len("trained_models/") :]

    # Construct the blob path and download the model
    blob = bucket.blob(f"trained_models/{model_name}")  # Ensure correct path in GCP bucket

    # Extract the model filename
    model_filename = model_name.split("/")[-1]
    local_path = LOCAL_MODEL_DIR / model_filename

    if not local_path.exists():
        print(f"Downloading model {model_name} from GCP bucket {BUCKET_NAME}...")
        try:
            blob.download_to_filename(local_path)
            print(f"Model downloaded to {local_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_name}: {str(e)}")
    else:
        print(f"Model {local_path} already exists locally. Skipping download.")

    return local_path


def load_model(model_path):
    """
    Load the model and assign as current model
    """
    global model, current_model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found.")

    model_name = model_path.stem.replace("trained_", "").split("-")[0]
    model = Model.load_from_checkpoint(
        model_path, model_name=model_name, num_classes=1, map_location=torch.device("cpu")
    )
    model.eval()
    current_model_name = model_name  # Update the global variable
    print(f"Loaded model: {current_model_name}")


async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI app
    """
    global model, current_model_name
    print("Initializing default model...")
    LOCAL_MODEL_DIR.mkdir(exist_ok=True)

    # Default model that gets loaded (from cloud)
    default_model_name = "trained_densenet121-v1.ckpt"
    try:
        # Check if the model exists in GCP before attempting to download
        available_models = list_models_in_gcp()
        if f"trained_models/{default_model_name}" not in available_models:
            raise RuntimeError(f"Model: {default_model_name} not found in the GCP bucket.")

        model_path = download_model_from_gcp(default_model_name)
        print(model_path)
        load_model(model_path)
        print(f"Default model - '{current_model_name}' loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the default model: {str(e)}")

    yield

    # Cleanup
    print("Cleaning up...")
    del model


app = FastAPI(lifespan=lifespan)


@app.get("/models")
async def list_models():
    """
    List available models in the GCP bucket.
    """
    available_models = list_models_in_gcp()
    return {"available_models": available_models}


# Helper for requesting the model
class ModelSelectRequest(BaseModel):
    model_name: str


@app.post("/models/select", response_model=dict)
async def select_model(request: ModelSelectRequest):
    """
    Select a specific model for inference from the GCP bucket.
    """
    global model, current_model_name  # updating

    # Print the request data for debugging
    print(f"Received request: {request.dict()}")
    model_name = request.model_name
    available_models = list_models_in_gcp()
    print(f"Available models: {available_models}")

    # Check if the requested model exists in GCP bucket
    if model_name not in available_models:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"Model '{model_name}' not found in the bucket.",
                "available_models": available_models,
            },
        )

    try:
        # Download and load the model
        model_path = download_model_from_gcp(model_name)
        load_model(model_path)
        current_model_name = model_name  # Update the current model name
        return {"message": f"Model '{model_name}' loaded successfully."}

    except Exception as e:
        # Return a more specific error if something goes wrong during model loading
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to load model '{model_name}': {str(e)}"},
        )


@app.get("/models/current")
async def get_current_model():
    """
    Get the name of the currently loaded model.
    """
    if current_model_name is None:
        raise HTTPException(status_code=404, detail="No model is currently loaded.")
    return {"current_model": current_model_name}


# Data drifting stuff


# Function to preprocess images and get CLIP embeddings
def get_clip_embeddings(images, clip_model, processor, batch_size=32):
    """Process images and extract CLIP embeddings."""
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float32) / 255.0
        if batch_tensor.ndim == 4 and batch_tensor.shape[1] == 3:
            batch_tensor = batch_tensor.permute(0, 2, 3, 1)

        # Convert to PIL (Processor expects PIL images)
        pil_images = [Image.fromarray((img.numpy() * 255).astype("uint8")) for img in batch_tensor]

        # Use CLIP processor for pre-mebeddings
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)

        with torch.no_grad():
            batch_embeddings = clip_model.get_image_features(inputs["pixel_values"])
        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)


# Generate CLIP embeddings for the training dataset
initialize_clip()  # Load CLIP model and processor
print("Getting training embeddings for drift detection...")

# Instead of computing embedding each time
embedding_file = "train_embeddings.pkl"
if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        train_embeddings = pickle.load(f)
    print("Loaded embeddings.")
else:
    train_images_path = project_root / "data" / "processed" / "train_images.pt"
    if not train_images_path.exists():
        raise FileNotFoundError(f"File {train_images_path} not found.")
    train_images = torch.load(train_images_path)
    train_images_np = train_images.cpu().numpy()
    train_embeddings = get_clip_embeddings(train_images_np, clip_model, processor)
    with open(embedding_file, "wb") as f:
        pickle.dump(train_embeddings, f)
    print("Computed and saved embeddings.")


# Create reference DataFrame for drift detection
reference_df = pd.DataFrame(train_embeddings, columns=[f"Feature_{i}" for i in range(train_embeddings.shape[1])])
reference_df["Dataset"] = "X-RAY"
print("Reference dataset created.")

# Initialize `current_df` as a copy of `reference_df`
current_df = reference_df.copy()
current_df["Dataset"] = "Current"


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <h1>Welcome to the Pneumonia Detection API</h1>
    <p>Use the following endpoints:</p>
    <ul>
        <li><a href="/docs">Interactive Docs (Swagger)</a></li>
        <li><a href="/monitoring">Monitoring Dashboard</a></li>
    </ul>
    """


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
        html_content = await f.read()  # Loads page (slowly)

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
        predicted = torch.sigmoid(output) > 0.5
        label = true_labels[predicted.item()]

    # Prepare the image for CLIP (no need to transform)
    inputs = processor(images=image_tensor, return_tensors="pt", padding=True, do_rescale=False)

    # Get embeddings from CLIP model
    with torch.no_grad():
        predicted_embedding = clip_model.get_image_features(inputs["pixel_values"])  # Correct method for CLIP model

    # Remove the extra dimensions (squeeze) and convert to numpy
    predicted_embedding = predicted_embedding.squeeze(0).numpy()  # (512,) embeddings

    # Convert the embeddings into a DataFrame
    new_embedding = pd.DataFrame(
        [predicted_embedding], columns=[f"Feature_{i}" for i in range(predicted_embedding.shape[0])]
    )  # predicted_embedding.shape[0] will give 512
    new_embedding["Dataset"] = "Current"

    # Add the new embedding to the current DataFrame
    global current_df
    current_df = pd.concat([current_df, new_embedding], ignore_index=True)

    return JSONResponse(content={"label": label})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.mlops_project.api:app", host="0.0.0.0", port=8000, reload=True)

    """
    Commands:
    # Get current model
    curl -X GET "http://localhost:8000/models/current"

    # Get current models in buckets and their link name.
    curl -X GET "http://localhost:8000/models"

    # Predict using model (example)
    curl -X POST "http://localhost:8000/predict_pneumonia" -H "Content-Type: multipart/form-data" -F "file=@C:\person1_virus_6.JPEG"

    # Switch to a different trained model from list of models in the online bucket (switches from simple to resnet50)
    curl -X POST "http://localhost:8000/models/select" -H "Content-Type: application/json" -d "{\"model_name\": \"trained_models/trained_resnet50-v7.ckpt\"}"

    # Monitoring for datadrift
    Opening in browser: http://localhost:8000/monitoring or go http://localhost:8000 and click monitoring dashboard.
    """
