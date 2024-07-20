from abc import ABC, abstractmethod
from src.utils import DataLoader, random_split
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import KFold


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


class KFoldDataLoaderFactory(DataLoaderFactory):
    """
    A factory class for creating data loaders for k-fold cross-validation.
    This class prepares the dataset for k-fold cross-validation and provides methods to create
    train and validation data loaders for each fold.
    """
    def __init__(self, dataset_strategy, transform, k_folds, batch_size, num_workers):
        """
        Initialize the KFoldDataLoaderFactory.

        Args:
            dataset_strategy: Strategy for creating the dataset.
            transform: Transformations to apply to the dataset.
            k_folds (int): Number of folds for cross-validation.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of worker processes for data loading.
        """
        self.dataset = dataset_strategy.create_dataset(transform)
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.current_fold = 0
        self.folds = None

    def prepare_folds(self):
        """
        Prepare the k-fold splits of the dataset.
        This method should be called before creating any data loaders.
        """
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        self.folds = list(kfold.split(self.dataset))

    def create_train_loader(self):
        """
        Create a DataLoader for the training data of the current fold.

        Returns:
            DataLoader: A DataLoader for the training data.
        """
        train_indices, _ = self.folds[self.current_fold]
        train_sampler = SubsetRandomSampler(train_indices)
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers)

    def create_val_loader(self):
        """
        Create a DataLoader for the validation data of the current fold.

        Returns:
            DataLoader: A DataLoader for the validation data.
        """
        _, val_indices = self.folds[self.current_fold]
        val_sampler = SubsetRandomSampler(val_indices)
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers=self.num_workers)

    def create_test_loader(self):
        """
        Create a DataLoader for the test data.
        In k-fold cross-validation, we typically don't have a separate test set.

        Returns:
            None: As there's no separate test set in k-fold cross-validation.
        """
        return None

    def next_fold(self):
        """
        Move to the next fold.
        This method should be called after completing training and validation on the current fold.
        """
        self.current_fold = (self.current_fold + 1) % self.k_folds
