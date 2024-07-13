from src.data import *


class VanderbiltHippocampusDataset(Dataset):
    """
    Dataset class for loading and preprocessing hippocampus MRI data.
    """

    def __init__(self, json_file=Config.VANDERBILT_DATA_JSON, root_dir=Config.VANDERBILT_DATA_DIR, transform=None):
        """
        Initialize the dataset.

        Args:
            json_file (str): Path to the JSON file containing dataset information.
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self._check_and_download_dataset(root_dir)
        self._load_json_data(json_file)
        self.root_dir = root_dir
        self.transform = transform if transform is not None else train_transform
        self._extract_image_and_label_paths()

    @staticmethod
    def _check_and_download_dataset(root_dir):
        if not os.path.isdir(root_dir):
            print("Dataset not found. Starting download...")
            download_dataset()

    def _load_json_data(self, json_file):
        with open(json_file, "r") as file:
            self.json_data = json.load(file)
        self.train_data = self.json_data["training"]

    def _extract_image_and_label_paths(self):
        self.images = [item["image"].lstrip('./') for item in self.train_data]
        self.labels = [item["label"].lstrip('./') for item in self.train_data]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetch a sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label) where image is a float tensor and label is a long tensor.
        """
        img_path = os.path.join(self.root_dir, self.images[idx])
        label_path = os.path.join(self.root_dir, self.labels[idx])

        image = self.load_and_preprocess(img_path)
        label = self.load_and_preprocess(label_path)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'].float(), sample['label'].long()

    @staticmethod
    def load_and_preprocess(file_path):
        """
        Load and preprocess a NIfTI file.

        Args:
            file_path (str): Path to the NIfTI file.

        Returns:
            torch.Tensor: Preprocessed data as a PyTorch tensor.
        """
        data = nib.load(file_path).get_fdata()
        data = torch.from_numpy(data).float()
        data = data.unsqueeze(0)  # Add channel dimension
        return data

    # Explanation: __getitem__ is crucial. It's called by the DataLoader to fetch


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
