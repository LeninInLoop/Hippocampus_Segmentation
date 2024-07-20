from abc import ABC, abstractmethod
from .Train import *
from ..data import DataSetLoader


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies.
    This class defines the interface for different training strategies (e.g., standard training, k-fold cross-validation).
    """
    def __init__(self, data_loader_factory):
        """
        Initialize the TrainingStrategy.

        Args:
            data_loader_factory: A factory object for creating data loaders.
        """
        self.data_loader_factory = data_loader_factory

    @abstractmethod
    def prepare_data(self):
        """Prepare the data for training. This method should be implemented by subclasses."""
        pass

    @abstractmethod
    def train(self, model, device, optimizer):
        """
        Train the model.

        Args:
            model: The model to train.
            device: The device to use for training (e.g., 'cpu' or 'cuda').
            optimizer: The optimizer to use for training.
        """
        pass

    @abstractmethod
    def validate(self, model, device):
        """
        Validate the model.

        Args:
            model: The model to validate.
            device: The device to use for validation.

        Returns:
            The validation result.
        """
        pass

    @abstractmethod
    def final_validation(self, model, device):
        """
        Perform final validation of the model.

        Args:
            model: The model to validate.
            device: The device to use for validation.

        Returns:
            The final validation result.
        """
        pass

    def execute(self, model, device, optimizer):
        """
        Execute the full training and validation process.

        Args:
            model: The model to train and validate.
            device: The device to use for training and validation.
            optimizer: The optimizer to use for training.

        Returns:
            A tuple containing the validation result and final validation result.
        """
        self.prepare_data()
        self.train(model, device, optimizer)
        val_result = self.validate(model, device)
        final_results = self.final_validation(model, device)
        return val_result, final_results

    def execute(self, model, device, optimizer):
        self.prepare_data()
        self.train(model, device, optimizer)
        val_result = self.validate(model, device)
        final_results = self.final_validation(model, device)
        return val_result, final_results


class StandardTrainingStrategy(TrainingStrategy):
    """
    A training strategy for standard (non-k-fold) training.
    This strategy uses separate train, validation, and test datasets.
    """
    def __init__(self, data_loader_factory):
        """
        Initialize the StandardTrainingStrategy.

        Args:
            data_loader_factory: A factory object for creating data loaders.
        """
        super().__init__(data_loader_factory)
        dataset = DataSetLoader(data_loader_factory)
        self.train_loader = dataset.train_loader
        self.val_loader = dataset.val_loader
        self.test_loader = dataset.test_loader

    def prepare_data(self):
        """No additional preparation needed for standard training."""
        pass

    def train(self, model, device, optimizer):
        """
        Train the model using the standard training approach.

        Args:
            model: The model to train.
            device: The device to use for training.
            optimizer: The optimizer to use for training.
        """
        trainer = Train(model, device, self.train_loader, self.val_loader, optimizer)
        trainer.start_training()

    def validate(self, model, device):
        """
        Validate the model using the validation dataset.

        Args:
            model: The model to validate.
            device: The device to use for validation.

        Returns:
            The validation result.
        """
        print("Validating Using validation Dataset:")
        return Validation.load_and_validate(
            model=model,
            model_path=Config.BEST_MODEL_SAVE_PATH,
            val_loader=self.val_loader,
            device=device
        )

    def final_validation(self, model, device):
        """
        Perform final validation of the model using the test dataset.

        Args:
            model: The model to validate.
            device: The device to use for validation.

        Returns:
            The final validation result.
        """
        print("Validating Using Test Dataset:")
        return Validation.load_and_validate(
            model=model,
            model_path=Config.BEST_MODEL_SAVE_PATH,
            val_loader=self.test_loader,
            device=device,
            is_test_dataset=True
        )


class KFoldTrainingStrategy(TrainingStrategy):
    """
    A training strategy for k-fold cross-validation.
    This strategy trains and validates the model k times, each time using a different fold as the validation set.
    """
    def __init__(self, data_loader_factory):
        """
        Initialize the KFoldTrainingStrategy.

        Args:
            data_loader_factory: A factory object for creating data loaders.
        """
        super().__init__(data_loader_factory)
        self.fold_results = []

    def prepare_data(self):
        """Prepare the data for k-fold cross-validation."""
        self.data_loader_factory.prepare_folds()

    def train(self, model, device, optimizer):
        """
        Train the model for the current fold.

        Args:
            model: The model to train.
            device: The device to use for training.
            optimizer: The optimizer to use for training.
        """
        train_loader = self.data_loader_factory.create_train_loader()
        val_loader = self.data_loader_factory.create_val_loader()
        trainer = Train(model, device, train_loader, val_loader, optimizer)
        trainer.start_training()

    def validate(self, model, device):
        """
        Validate the model for the current fold.

        Args:
            model: The model to validate.
            device: The device to use for validation.

        Returns:
            The validation result for the current fold.
        """
        print(f"Validating fold {self.data_loader_factory.current_fold + 1}/{self.data_loader_factory.k_folds}")
        val_loader = self.data_loader_factory.create_val_loader()
        result = Validation.load_and_validate(
            model=model,
            model_path=Config.BEST_MODEL_SAVE_PATH,
            val_loader=val_loader,
            device=device
        )
        self.fold_results.append(result)
        return result

    def final_validation(self, model, device):
        """
        Perform final validation by averaging results across all folds.

        Args:
            model: The model to validate.
            device: The device to use for validation.

        Returns:
            The average validation result across all folds.
        """
        print("Final validation using average of fold validations:")
        return sum(self.fold_results) / len(self.fold_results)

    def execute(self, model, device, optimizer):
        """
        Execute the full k-fold cross-validation process.

        Args:
            model: The model to train and validate.
            device: The device to use for training and validation.
            optimizer: The optimizer to use for training.

        Returns:
            A tuple containing the list of fold results and the final average result.
        """
        self.prepare_data()
        for _ in range(self.data_loader_factory.k_folds):
            self.train(model, device, optimizer)
            self.validate(model, device)
            self.data_loader_factory.next_fold()
        return self.fold_results, self.final_validation(model, device)
