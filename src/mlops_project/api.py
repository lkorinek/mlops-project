from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from src.mlops_project.model import Model
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Project root to get the models path
project_root = Path(__file__).resolve().parents[2]

async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI app by loading and cleaning up the model.
    """
    global model
    model_path = "trained_resnet50" + ".ckpt" # must be densenet, resnet, vgg16 or simple (also takes versions)
    print("Loading model...")
    trained_model_path = project_root / "models" / model_path

    # Ensure the model file exists
    if trained_model_path.exists() and trained_model_path.suffix == ".ckpt":
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

app = FastAPI(lifespan=lifespan)

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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

    # Transform the image for the model
    image_tensor = transform(image).unsqueeze(0)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        label = true_labels[predicted.item()]
    return JSONResponse(content={"label": label})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.mlops_project.api:app", host="0.0.0.0", port=8000, reload=True)


    # Disclaimer: I cannot get it to work by just running the file
    # I can however get it to run by running CLI: uvicorn src.mlops_project.api:app --host 0.0.0.0 --port 8000
    # And then test with an image like so: 
    # curl -X POST "http://localhost:8000/predict_pneumonia" -H "Content-Type: multipart/form-data" -F "file=@C:\person1_virus_6.JPEG"
    # In a terminal windows. I got the following output: {"label":"Normal"} 
    # TODO typer implementation so we can choose the model from CLI and maybe path to image we wish to predict on.