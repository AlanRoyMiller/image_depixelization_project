
import torch
from torchvision import transforms
from matplotlib import pyplot as plt

def visualize_predictions(network: torch.nn.Module, test_loader: torch.utils.data.DataLoader, output_dir: str):
    """Function to visualize predictions made by the trained model.
    
    This function takes a trained neural network and a DataLoader containing the test data,
    and creates visualizations showing the original pixelated images and their corresponding 
    predicted depixelized versions side by side. The visualizations are saved to the specified output directory.
    
    Args:
        network (torch.nn.Module): The trained neural network.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test data.
        output_dir (str): Directory to save the visualization images.
    
    Returns:
        None
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
    ])
    
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(test_loader):
            input_tensor = input_tensor.to(device)
            output = network(input_tensor)
            
            # Convert tensors to images
            input_image = transform(input_tensor[0].cpu())
            output_image = transform(output[0].cpu())
            
            # Create a new figure
            plt.figure()
            
            # Plot the original pixelated image
            plt.subplot(1, 2, 1)
            plt.imshow(input_image, cmap='gray')
            plt.title('Original Pixelated Image')
            
            # Plot the predicted depixelized image
            plt.subplot(1, 2, 2)
            plt.imshow(output_image, cmap='gray')
            plt.title('Predicted Depixelized Image')
            
            # Save the visualization to the output directory
            plt.savefig(f"{output_dir}/visualization_{i+1}.png")
            plt.close()

if __name__ == "__main__":
    # Example usage (this part will be removed in the final version)
    pass
