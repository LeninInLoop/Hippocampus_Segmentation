from src.data import *


class HippocampusDataset(Dataset):
    def __init__(self, json_file=Config.DATA_JSON, root_dir=Config.DATA_DIR, transform=None):
        if not os.path.isdir(root_dir):
            print("dataset not found: Start Downloading....")
            download_dataset()
        with open(json_file, "r") as file:
            self.json_data = json.load(file)
        self.train_data = self.json_data["training"]
        self.root_dir = root_dir
        self.transform = transform
        self.images = [item["image"].lstrip('./') for item in self.train_data]
        self.labels = [item["label"].lstrip('./') for item in self.train_data]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        label_path = os.path.join(self.root_dir, self.labels[idx])

        image = self.load_and_preprocess(img_path)
        label = self.load_and_preprocess(label_path, is_label=True)

        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image).float(), torch.from_numpy(label).long()

    def load_and_preprocess(self, file_path, is_label=False):
        data = nib.load(file_path).get_fdata()
        padded_data = pad_to_size(data)

        if not is_label:
            padded_data = self.normalize(padded_data)
            padded_data = np.expand_dims(padded_data, axis=0)  # Add channel dimension

        return padded_data

    @staticmethod
    def normalize(image):
        return (image - image.min()) / (image.max() - image.min())

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
