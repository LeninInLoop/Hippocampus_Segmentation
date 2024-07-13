from abc import ABC, abstractmethod
from src.utils import DataLoader, random_split


class DataLoaderFactory(ABC):
    @abstractmethod
    def create_train_loader(self):
        """
        Abstract method to create a training data loader.
        This method should be implemented by all concrete factory classes.

        Returns:
            A DataLoader object for the training dataset.
        """
        pass

    @abstractmethod
    def create_val_loader(self):
        """
        Abstract method to create a validation data loader.
        This method should be implemented by all concrete factory classes.

        Returns:
            A DataLoader object for the validation dataset.
        """
        pass


class DefaultDataLoaderFactory(DataLoaderFactory):
    def __init__(self, dataset_strategy, transform, split_ratio, batch_size, num_workers):
        """
        Initialize the DefaultDataLoaderFactory.

        Args:
            dataset_strategy: A strategy object for creating the dataset.
            transform: The data transformation to be applied to the dataset.
            split_ratio (float): The ratio of the dataset to use for training (0.0 to 1.0).
            batch_size (int): The number of samples per batch.
            num_workers (int): The number of subprocesses to use for data loading.
        """
        self.dataset_strategy = dataset_strategy
        self.transform = transform
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.full_dataset = self.dataset_strategy.create_dataset(self.transform)
        self.train_dataset, self.val_dataset = self._split_dataset()

    def _split_dataset(self):
        """
        Split the full dataset into training and validation sets.

        Returns:
            A tuple containing the training and validation datasets.
        """
        total_size = len(self.full_dataset)
        train_size = int(self.split_ratio * total_size)
        val_size = total_size - train_size
        return random_split(self.full_dataset, [train_size, val_size])

    def _create_data_loader(self, dataset, shuffle):
        """
        Create a DataLoader with the specified parameters.

        Args:
            dataset: The dataset to load data from.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            A DataLoader object.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers
        )

    def create_train_loader(self):
        """
        Create a DataLoader for the training dataset.

        Returns:
            A DataLoader object for the training dataset.
        """
        return self._create_data_loader(self.train_dataset, shuffle=True)

    def create_val_loader(self):
        """
        Create a DataLoader for the validation dataset.

        Returns:
            A DataLoader object for the validation dataset.
        """
        return self._create_data_loader(self.val_dataset, shuffle=False)
