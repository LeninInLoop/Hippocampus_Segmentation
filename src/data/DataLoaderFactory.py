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

    @abstractmethod
    def create_test_loader(self):
        """
        Create a DataLoader for the test dataset.

        Returns:
            A DataLoader object for the validation dataset.
        """
        pass


class DefaultDataLoaderFactory(DataLoaderFactory):
    def __init__(self, dataset_strategy, transform, train_ratio, val_ratio, test_ratio, batch_size, num_workers):
        """
        Initialize the DefaultDataLoaderFactory.

        Args:
            dataset_strategy: A strategy object for creating the dataset.
            transform: The data transformation to be applied to the dataset.
            train_ratio (float): The ratio of the dataset to use for training (e.g., 0.6 for 60%).
            val_ratio (float): The ratio of the dataset to use for validation (e.g., 0.1 for 10%).
            test_ratio (float): The ratio of the dataset to use for testing (e.g., 0.3 for 30%).
            batch_size (int): The number of samples per batch.
            num_workers (int): The number of subprocesses to use for data loading.
        """
        self.dataset_strategy = dataset_strategy
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.full_dataset = self.dataset_strategy.create_dataset(self.transform)
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset(self.full_dataset)

    def _split_dataset(self, dataset):
        """
        Split the full dataset into training, validation, and test sets.

        Returns:
            A tuple containing the training, validation, and test datasets.
        """
        total_size = len(dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        return random_split(dataset, [train_size, val_size, test_size])

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
        return self._create_data_loader(self.val_dataset, shuffle=True)

    def create_test_loader(self):
        """
        Create a DataLoader for the test dataset.

        Returns:
            A DataLoader object for the validation dataset.
        """
        return self._create_data_loader(self.test_dataset, shuffle=True)
