from torch.utils.data import random_split, DataLoader
from src.Config import Config
from .DataSet import HippocampusDataset


class DataSetLoader:
    def __init__(self):
        self.full_dataset = HippocampusDataset()
        self.train_dataset, self.val_dataset = self._split_dataset()
        self.train_loader = self._create_data_loader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_data_loader(self.val_dataset, shuffle=False)

    # Explanation: The __init__ method sets up the entire data pipeline:
    # 1. It creates the full dataset
    # 2. Splits it into training and validation sets
    # 3. Creates DataLoaders for both sets
    # This encapsulation makes it easy to manage the entire data loading process.

    def _split_dataset(self):
        total_size = len(self.full_dataset)
        train_size = int(Config.TRAIN_SPLIT_RATIO * total_size)
        val_size = total_size - train_size
        return random_split(self.full_dataset, [train_size, val_size])

    # Explanation: This method splits the dataset into training and validation sets.
    # It uses PyTorch's random_split function, which ensures a random division of the data.
    # The split ratio is defined in the Config, allowing for easy adjustment.

    @staticmethod
    def _create_data_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS
        )

    # Explanation: This method creates a DataLoader for a given dataset.
    # - It uses the batch size and number of workers defined in Config.
    # - Shuffling is optional, typically used for the training set but not for validation.

    # Comparison:
    # Using DataLoader vs. manual batching:
    # Advantages of DataLoader:
    # - Efficient data loading with multi-processing
    # - Automatic batching and shuffling
    # - Easily configurable (e.g., batch size, shuffling, num_workers)
    # Disadvantages:
    # - Slight overhead for very small datasets
    # - May require careful tuning of num_workers for optimal performance

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    # Explanation: These getter methods provide access to the train and validation loaders.
    # They encapsulate the data loading process, allowing other parts of the code to easily
    # access the prepared data without needing to know the details of how it was set up.

# Overall explanation:
# This DataSetLoader class provides a clean, encapsulated way to handle the entire
# data loading process for a deep learning model. It separates the concerns of
# dataset creation, splitting, and loading, making the code more modular and easier to maintain.

# The use of a Config class for parameters like batch size and split ratio
# makes it easy to adjust these values without changing the core logic.
