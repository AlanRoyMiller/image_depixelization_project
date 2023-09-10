import numpy as np


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:

    """
    to_grayscale takes in a numpy array representing an image in RGB format
    and returns a numpy array representing the same image but in grayscale format with an additional brightness channel.
    This function is based on the colorimetric conversion.
    See:  https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale


    """

    if len(pil_image.shape) == 2:
        pil_image_copy = pil_image.copy()
        np.expand_dims(pil_image_copy, axis=0)
        return pil_image_copy

    elif pil_image.shape[2] != 3:
        raise ValueError

    pil_image_copy = pil_image.copy() / 255.0

    c_linear = np.where((pil_image_copy <= 0.04045), (pil_image_copy / 12.92),
                        (((pil_image_copy + 0.055) / 1.055) ** 2.4))

    y_linear = c_linear[:, :, 0] * 0.2126 + c_linear[:, :, 1] * 0.7152 + c_linear[:, :, 2] * 0.0722

    y_srgb = np.where((y_linear <= 0.0031308), (y_linear * 12.92), (((y_linear ** (1 / 2.4)) * 1.055) - 0.055))

    pil_image_copy = (y_srgb * 255.0).astype(pil_image.dtype)

    if np.issubdtype(pil_image.dtype, np.integer):
        pil_image_copy = np.round(pil_image_copy)

    pil_image_copy = np.expand_dims(pil_image_copy, axis=0)

    return pil_image_copy
