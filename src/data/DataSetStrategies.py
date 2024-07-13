from abc import ABC, abstractmethod
from src.data import VanderbiltHippocampusDataset


# Abstract base class for dataset strategies
class DatasetStrategy(ABC):
    @abstractmethod
    def create_dataset(self, transform):
        """
        Abstract method to create a dataset.
        This method should be implemented by all concrete strategy classes.

        Args:
            transform: The data transformation to be applied to the dataset.

        Returns:
            A dataset object.
        """
        pass


# Concrete strategy for the Vanderbilt Hippocampus dataset
class VanderbiltHippocampusDatasetStrategy(DatasetStrategy):
    def create_dataset(self, transform):
        """
        Creates and returns a VanderbiltHippocampusDataset instance.

        Args:
            transform: The data transformation to be applied to the dataset.

        Returns:
            VanderbiltHippocampusDataset: An instance of the Vanderbilt Hippocampus dataset.
        """
        return VanderbiltHippocampusDataset(transform=transform)


# Placeholder for other dataset strategies
class OtherDatasetStrategy(DatasetStrategy):
    def create_dataset(self, transform):
        """
        Placeholder method for creating other types of datasets.
        This should be implemented when adding support for new datasets.

        Args:
            transform: The data transformation to be applied to the dataset.

        Returns:
            Should return an instance of the specific dataset.
        """
        pass
