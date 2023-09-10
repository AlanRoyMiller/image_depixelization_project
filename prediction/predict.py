import torch
import numpy as np
from models.simple_cnn import SimpleCNN

def predict(network: torch.nn.Module, test_loader: torch.utils.data.DataLoader, amount_of_predictions) -> list:
    # Function to perform predictions on new data using the trained model.

    network.eval()  # Set the network to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    predictions = []

    test_loader_length = len(test_loader.dataset)
    message = f"Making {min(test_loader_length, amount_of_predictions)} predictions"
    if amount_of_predictions > test_loader_length:
        message += f" instead of {amount_of_predictions} due to test loader being of shorter size."
    print(message)

    with torch.no_grad():  # Disable gradient computation
        prediction_count = 0
        for batch in test_loader:
            if prediction_count >= amount_of_predictions:
                break

            pixelated_images, known_arrays = batch
            pixelated_images = pixelated_images.to(device) 
            outputs = network(pixelated_images)

            # Convert the outputs to numpy arrays and flatten them
            outputs = outputs.cpu().numpy()
            known_arrays = known_arrays.cpu().numpy()

            for i in range(len(outputs)):
                if prediction_count >= amount_of_predictions:
                    break

                prediction = outputs[i][~known_arrays[i].astype(np.bool_)]
                predictions.append(prediction.astype(np.uint8))

                prediction_count += 1

    return predictions




    