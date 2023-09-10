import numpy as np



def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int,
                  size: int):  # -> tuple[np.ndarray, np.ndarray, np.ndarray]

    """
    Prepare an image by pixelating a specified area and generating a boolean mask for that area.

    Parameters
    ----------
    image : np.ndarray
        A single-channel input image as a NumPy array with shape (1, height, width).
    x : int
        The x-coordinate of the top-left corner of the area to be pixelated.
    y : int
        The y-coordinate of the top-left corner of the area to be pixelated.
    width : int
        The width of the area to be pixelated.
    height : int
        The height of the area to be pixelated.
    size : int
        The size of the square blocks used for pixelation.

    Returns
    -------
    pixelated_image : np.ndarray
        A modified version of the input image with the specified area pixelated.
    target_array : np.ndarray
        A NumPy array representing the original specified area in the input image.
    known_array : np.ndarray
        A boolean mask of the same shape as the input image, with `False` values at the pixelated area.

    Raises
    ------
    ValueError
        If the input image does not have the correct dimensions, shape, or if the input parameters are out of bounds.
    """

    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("Image should have 3 dimensions and a shape of (1, height, width)")

    if width < 2 or height < 2 or size < 2:
        raise ValueError("Width, height, and size must be greater than or equal to 2")

    if x < 0 or x + width > image.shape[2]:
        raise ValueError(f"Invalid x-coordinate. x should be in the range [0, {image.shape[2] - width}]")

    if y < 0 or y + height > image.shape[1]:
        raise ValueError(f"Invalid y-coordinate. y should be in the range [0, {image.shape[1] - height}]")

    pixelated_image = image.copy()
    pixelated_area = pixelated_image[:, y:y + height, x:x + width]

    for h in range(0, height, size):
        for w in range(0, width, size):
            mean = np.mean(pixelated_area[0, h: h + size, w: w + size])
            pixelated_area[0, h: h + size, w: w + size] = mean

    pixelated_image[:, y:y + height, x:x + width] = pixelated_area

    known_array = np.ones_like(image, dtype=bool)
    known_array[:, y:y + height, x:x + width] = False

    target_array = image[:, y:y + height, x:x + width]

    return pixelated_image, known_array, target_array
