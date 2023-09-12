from torchvision import transforms
from PIL import Image
import random
import numpy as np

def random_augmented_image(
    image,
    image_size,
    seed
): # -> np.ndarray

    np.random.seed(seed)
    
    transformers = [
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
    ]
    
    random_transforms = random.sample(transformers, 2)
    
    transform_chain = transforms.Compose([
        transforms.Resize(size=image_size),
        random_transforms[0],
        random_transforms[1],
        transforms.ToTensor(),
    ])
    
    transformed_image = transform_chain(image)
    transformed_image = transformed_image.permute(1, 2, 0).numpy()
    
    return transformed_image
