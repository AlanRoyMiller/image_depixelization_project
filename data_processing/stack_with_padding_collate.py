import torch
import numpy as np

def stack_with_padding(batch_as_list: list):
    max_height = max([sample[0].shape[1] for sample in batch_as_list])
    max_width = max([sample[0].shape[2] for sample in batch_as_list])
    
    stacked_pixelated_images = []
    stacked_known_arrays = []
    target_arrays = []
    image_files = []

    for pix, known_arr, target_arr, img_file in batch_as_list:
        padded_pix = np.pad(pix, pad_width=((0, 0), (0, max_height - pix.shape[1]), (0, max_width - pix.shape[2])),
                              mode="constant", constant_values=0)
        stacked_pixelated_images.append(padded_pix)

        padded_known = np.pad(known_arr, pad_width=((0, 0), (0, max_height - pix.shape[1]), (0, max_width - known_arr.shape[2])),
                              mode="constant", constant_values=1)
        stacked_known_arrays.append(padded_known)

        padded_target = np.pad(target_arr, pad_width=((0, 0), (0, max_height - target_arr.shape[1]), (0, max_width - target_arr.shape[2])),
                       mode="constant", constant_values=0)

        target_arr_torch = torch.from_numpy(padded_target)
        target_arrays.append(target_arr_torch)

        image_files.append(img_file)

    stacked_pixelated_images = torch.from_numpy(np.array(stacked_pixelated_images))
    stacked_known_arrays = torch.from_numpy(np.array(stacked_known_arrays))

    return stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files
