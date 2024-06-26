import os
import json
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader


class HippocampusDataset(Dataset):
    def __init__(self, data_dir, json_file, transform=None, mode='train'):
        """
        Initializes the HippocampusDataset.

        This constructor sets up the dataset by loading the JSON file and preparing
        the data list based on the specified mode (train or test).

        Args:
            data_dir (str): Path to the directory containing the dataset
            json_file (str): Name of the JSON file with dataset information
            transform (callable, optional): Optional transform to be applied on a sample
            mode (str): 'train' for training data, 'test' for test data

        Raises:
            ValueError: If mode is neither 'train' nor 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        # Load the JSON file
        with open(os.path.join(data_dir, json_file), 'r') as f:
            self.data_json = json.load(f)

        if mode == 'train':
            self.data_list = self.data_json['training']
        elif mode == 'test':
            self.data_list = self.data_json['test']
        else:
            raise ValueError("Mode must be either 'train' or 'test'")

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        This method is required for PyTorch datasets and is used by DataLoader
        to determine the number of batches.

        Returns:
            int: The total number of samples in the dataset
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Fetches and processes a single sample from the dataset.

        This method loads an image (and label for training mode) from the disk,
        applies necessary preprocessing, and returns the data in the format
        required by the model.

        Args:
            idx (int): Index of the sample to fetch

        Returns:
            tuple: (image, label) for training mode, (image,) for test mode
                   image is a normalized 3D tensor, label is a 3D tensor of integers
        """
        if self.mode == 'train':
            img_path = os.path.join(self.data_dir, self.data_list[idx]['image'][2:])  # Remove './' from the start
            label_path = os.path.join(self.data_dir, self.data_list[idx]['label'][2:])  # Remove './' from the start
        else:  # test mode
            img_path = os.path.join(self.data_dir, self.data_list[idx][2:])  # Remove './' from the start
            label_path = None

        # Load image
        image = nib.load(img_path).get_fdata()

        # Normalize image (assuming MRI intensity values)
        image = (image - image.min()) / (image.max() - image.min())

        # Convert image to torch tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            # Load and process label
            label = nib.load(label_path).get_fdata()
            label = torch.from_numpy(label).long()
            return image, label
        else:
            # For test mode, return only the image
            return image

    @staticmethod
    def get_data_loaders(data_dir, json_file, batch_size=4, num_workers=4):
        """
        Creates DataLoader objects for training, validation, and testing.

        This method sets up the complete data pipeline, including:
        - Creating separate datasets for training and testing
        - Splitting the training data into training and validation subsets
        - Creating DataLoader objects with specified batch size and number of workers

        Args:
            data_dir (str): Path to the directory containing the dataset
            json_file (str): Name of the JSON file with dataset information
            batch_size (int): Number of samples per batch
            num_workers (int): Number of subprocesses to use for data loading

        Returns:
            tuple: (train_loader, val_loader, test_loader)
                   Each loader is a PyTorch DataLoader object
        """
        train_dataset = HippocampusDataset(data_dir, json_file, mode='train')

        # Split into train and validation sets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Create test loader
        test_dataset = HippocampusDataset(data_dir, json_file, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader


data_dir = r'C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset'
json_file = r'C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset\dataset.json'

train_loader, val_loader, test_loader = HippocampusDataset.get_data_loaders(data_dir, json_file)
