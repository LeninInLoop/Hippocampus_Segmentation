class DataSetLoader:
    """
    A class to manage data loaders for training, validation and test datasets.

    This class uses a DataLoaderFactory to create and store data loaders
    for both training and validation datasets.
    """

    def __init__(self, data_loader_factory):
        """
        Initialize the DataSetLoader with a data loader factory.

        Args:
            data_loader_factory (DataLoaderFactory): A factory object that creates data loaders.
        """
        self.data_loader_factory = data_loader_factory
        self.train_loader = self.data_loader_factory.create_train_loader()
        self.val_loader = self.data_loader_factory.create_val_loader()
        self.test_loader = self.data_loader_factory.create_test_loader()

    def get_train_loader(self):
        """
        Get the data loader for the training dataset.

        Returns:
            DataLoader: The data loader for the training dataset.
        """
        return self.train_loader

    def get_val_loader(self):
        """
        Get the data loader for the validation dataset.

        Returns:
            DataLoader: The data loader for the validation dataset.
        """
        return self.val_loader

    def get_test_loader(self):
        """
        Get the data loader for the test dataset.

        Returns:
            DataLoader: The data loader for the validation dataset.
        """
        return self.test_loader
