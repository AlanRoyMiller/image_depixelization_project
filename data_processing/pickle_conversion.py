import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        self.pixelated_images = data['pixelated_images']
        self.known_arrays = data['known_arrays']

    def __len__(self):
        return len(self.pixelated_images)

    def __getitem__(self, idx):
        pixelated_image = self.pixelated_images[idx]
        known_array = self.known_arrays[idx]
        
        # Convert the numpy arrays to torch tensors
        pixelated_image = torch.tensor(pixelated_image, dtype=torch.float32)
        known_array = torch.tensor(known_array, dtype=torch.float32)

        
        
        return pixelated_image, known_array


