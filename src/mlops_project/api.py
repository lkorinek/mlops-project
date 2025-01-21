from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from model import Model
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

# Initialize FastAPI app
#app = FastAPI()

# Global variable to hold the model
model = None

async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI app by loading and cleaning up the model.

    Args:
        app (FastAPI): The FastAPI application instance.
    
    Yields:
        None: Used for the app's lifespan management.
    """
    global model
    print("Loading model")
    trained_model_path = "model_path" # my example path r"./models/trained_densenet121.ckpt"
    model_filename = Path(trained_model_path).name
    model_name = model_filename.replace("trained_", "").split('.')[0]
    if os.path.exists(trained_model_path) and trained_model_path.endswith('.ckpt'):
        # Create specific model with random weights
        model = Model(model_name=model_name, num_classes=1)
        checkpoint = torch.load(trained_model_path) # load checkpoint
        model.load_state_dict(checkpoint['state_dict']) # load the checkpoint weights
        model.eval()  # Set the model to evaluation mode
    else:
        raise HTTPException(status_code=400, detail="Model not found")
    
    yield  # Yield control back to FastAPI

    # Cleanup resources after shutdown
    print("Cleaning up")
    del model

#app.lifespan(lifespan)
app = FastAPI(lifespan=lifespan)

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the required input size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Define class labels
true_labels = ["Normal", "Pneumonia"]

@app.post("/predict_pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Predicts whether an uploaded chest X-ray image indicates pneumonia or normal.

    Args:
        file (UploadFile): The uploaded image file.
    
    Returns:
        JSONResponse: A JSON response with the predicted label.
    """
    # Open the uploaded image file
    image = Image.open(file.file)

    # Check if the image is in RGB format
    if image.mode != "RGB":
        raise HTTPException(status_code=400, detail="Image must be in RGB format")

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Predict the label
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = true_labels[predicted.item()]  # Map numerical label to string

    return JSONResponse(content={"label": label})

# Entry point for running the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)