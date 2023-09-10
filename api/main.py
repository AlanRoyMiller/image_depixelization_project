
from fastapi import FastAPI, File, UploadFile, HTTPException
from model.simple_cnn import SimpleCNN
from torchvision import transforms
from PIL import Image
import torch
import io

app = FastAPI()

# Define paths
model_save_path = "path/to/save/your/model.pth"  # Replace with the actual path to your trained model

# Load the trained model
network = SimpleCNN(1, 32, 3, True, 10)
network.load_state_dict(torch.load(model_save_path))
network.eval()

# Define the transform to convert tensors to images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
])

@app.post("/depixelize/")
async def depixelize(file: UploadFile):
    try:
        # Read the uploaded file
        image_data = await file.read()
        
        # Convert the uploaded file data to a PIL image
        image = Image.open(io.BytesIO(image_data)).convert('L')
        
        # Convert the PIL image to a tensor
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        
        # Move the tensor to the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        
        # Use the trained model to depixelize the image
        output_tensor = network(image_tensor)
        
        # Convert the output tensor to a PIL image
        output_image = transform(output_tensor[0].cpu())
        
        # Save the output image to a BytesIO object
        output_image_data = io.BytesIO()
        output_image.save(output_image_data, format='PNG')
        
        # Get the byte data from the BytesIO object
        output_image_data = output_image_data.getvalue()
        
        # Return the output image data as the response
        return {"image": output_image_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
