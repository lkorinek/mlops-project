from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image


# Initialize FastAPI app
app = FastAPI()

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

    # Load the trained model (update the path accordingly)
    trained_model_path = "path_for_the_model"  # TODO: Double-check and update this path
    if os.path.exists(trained_model_path) and trained_model_path.endswith('.ckpt'):
        print(f"Loading model from {trained_model_path}")
        model = Model.load_from_checkpoint(trained_model_path)
        model.eval()  # Set the model to evaluation mode
    else:
        raise HTTPException(status_code=400, detail="Model not found")
    
    yield  # Yield control back to FastAPI

    # Cleanup resources after shutdown
    print("Cleaning up")
    del model

# Attach the lifespan function to the FastAPI app
app.lifespan(lifespan)

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



                            