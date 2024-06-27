import json
import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
from src.Config import Config
import numpy as np


def pad_to_size(image, target_shape=(64, 64, 64)):
    # Get the current shape
    current_shape = image.shape

    # Calculate padding
    pad_width = [(0, max(target_shape[i] - current_shape[i], 0)) for i in range(3)]

    # Pad the image
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    # Crop if necessary (in case the original image is larger than target in any dimension)
    cropped_image = padded_image[:target_shape[0], :target_shape[1], :target_shape[2]]

    return cropped_image


# Explanation: This function pads (or crops) an input image to a target shape.
# It's crucial for ensuring all images have the same dimensions, which is
# necessary for batch processing in neural networks. The padding is done with
# zeros, and if the original image is larger, it's cropped to fit.

class HippocampusDataset(Dataset):
    def __init__(self, json_file=Config.DATA_JSON, root_dir=Config.DATA_DIR, transform=None):
        with open(json_file, "r") as file:
            self.json_data = json.load(file)

        self.train_data = self.json_data["training"]
        self.root_dir = root_dir
        self.transform = transform

        self.images = [item["image"][2:] for item in self.train_data]
        self.labels = [item["label"][2:] for item in self.train_data]

    # Explanation: The __init__ method sets up the dataset. It loads the JSON file
    # containing image and label paths, and stores these paths for later use.
    # The [2:] slicing removes the './' prefix from the file paths if present.

    def __len__(self):
        return len(self.images)

    # Explanation: __len__ returns the number of samples in the dataset.
    # This is used by PyTorch's DataLoader to know how many batches to create.

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        label_path = os.path.join(self.root_dir, self.labels[idx])

        # Load image and label
        image = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Pad (and potentially crop) image and label
        image_padded = pad_to_size(image)
        label_padded = pad_to_size(label)

        # Normalize image
        image_padded = (image_padded - image_padded.min()) / (image_padded.max() - image_padded.min())

        # Add channel dimension
        image_padded = np.expand_dims(image_padded, axis=0)

        if self.transform:
            image_padded = self.transform(image_padded)

        return torch.from_numpy(image_padded).float(), torch.from_numpy(label_padded).long()

    # Explanation: __getitem__ is crucial. It's called by the DataLoader to fetch
    # individual samples. Here's what it does:
    # 1. Loads the image and label from files
    # 2. Pads both to a standard size
    # 3. Normalizes the image to [0, 1] range
    # 4. Adds a channel dimension to the image (making it 1x64x64x64)
    # 5. Applies any additional transforms
    # 6. Converts both image and label to PyTorch tensors
    # The image is returned as a float tensor, while the label is a long tensor,
    # which is typical for segmentation tasks.


"""
Padding to a fixed size (e.g., 1x64x64x64):
Advantages:
- Preserves original voxel intensities, which can be crucial in medical imaging.
- Maintains spatial relationships within the original image.
- No interpolation artifacts.
- Better for cases where exact voxel values are important for diagnosis or segmentation.

Disadvantages:
- May introduce artificial boundaries due to padding.
- Can lead to loss of information if original images are larger and get cropped.
- Might result in inefficient use of the tensor if many images are much smaller than the target size.

Resizing with interpolation:
Advantages:
- Ensures all spatial information is included, just at a different scale.
- Can handle varying input sizes more gracefully.
- Often results in more efficient use of the tensor space.
- May generalize better if the model needs to work with images of different scales.

Disadvantages:
- Changes the original voxel intensities.
- Can introduce interpolation artifacts.
- Might alter fine details that could be important in medical imaging.
"""

# Explanation: This comment block provides a comparison between padding and resizing
# approaches for handling varying image sizes. It's important to understand these
# trade-offs when deciding on a preprocessing strategy for medical imaging tasks.
# In this implementation, padding is chosen, likely due to the importance of
# preserving original voxel intensities and spatial relationships in medical imaging.
