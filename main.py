
import torch
from torch.utils.data import DataLoader
from training.training_loop import training_loop
from prediction.predict import predict
from prediction.visualize_predictions import visualize_predictions
from data_processing.random_image_pixelation_dataset import RandomImagePixelationDataset
from data_processing.stack_with_padding_collate import stack_with_padding
from data_processing.pickle_conversion import CustomDataset
from models.simple_cnn import SimpleCNN
import os

def main():
    # Define paths and parameters
    train_data_path = "Image_files/images"  # Replace with the actual path to your training data
    test_data_pickle_path = "Image_files/test_set.pkl"  # Replace with the actual path to your test data pickle file
    output_dirs = ["outputs/visualizations", "outputs/models"]  # Replace with the desired path to save your visualizations

    # Create output directory if it does not exist
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    #train model if it does not exist

    

    # Set up data loaders
    train_dataset = RandomImagePixelationDataset(
        train_data_path, 
        width_range=(4, 32), 
        height_range=(4, 32), 
        size_range=(4, 16)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=stack_with_padding, shuffle=True)
    test_dataset = CustomDataset(test_data_pickle_path)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Initialize the model
    network = SimpleCNN(1, 32, 3, True, 10)

    # Train the model
    if not os.path.exists(f"{output_dirs[1]}/model.pth"):
        training_loop(f"{output_dirs[1]}/model.pth", network, train_loader, test_loader, num_epochs=15)
        print("Training complete. Open outputs folder to view results.")

    # Load the trained model
    print("Loading model and making predictions...")
    network.load_state_dict(torch.load(f"{output_dirs[1]}/model.pth"))

    # Make predictions on new data
    predictions = predict(network, test_loader, 10)

    # Visualize the predictions
    visualize_predictions(network, test_loader, output_dirs[0])

if __name__ == "__main__":
    main()
