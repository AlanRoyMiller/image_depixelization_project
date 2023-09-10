from torchvision import transforms
from PIL import Image
import random
import torch


def random_augmented_image(
    image,
    image_size,
    seed
): # -> torch.Tensor

    torch.random.manual_seed(seed)

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
    torch.nn.Dropout(p=0.1)
])

    transformed_image = transform_chain(image)

    return transformed_image

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    image_path = "path/to/File.jpg"
    with Image.open(image_path) as image:
        transformed_image = random_augmented_image(image, image_size=300, seed=3)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(transforms.functional.to_pil_image(transformed_image))
        axes[1].set_title("Transformed image")
        fig.tight_layout()
        plt.show()
