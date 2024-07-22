from abc import ABC, abstractmethod
from .Train import *
from ..data import DataSetLoader
from .Validate import Validation
from typing import List, Dict, Union


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
        trainer = Train(model, device, self.train_loader, self.val_loader, optimizer, output_dir=Config.LOGS_FOLDER)
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
        for fold in range(self.data_loader_factory.k_folds):
            print()
            print("\n" + "=" * 50)
            print()
            print(f"Fold {self.data_loader_factory.current_fold + 1}/{self.data_loader_factory.k_folds}")
            print("\n" + "=" * 50)
            fold_dir = os.path.join(Config.LOGS_FOLDER, f'fold{fold + 1}')
            os.makedirs(fold_dir, exist_ok=True)

            train_loader = self.data_loader_factory.create_train_loader()
            val_loader = self.data_loader_factory.create_val_loader()

            trainer = Train(model, device, train_loader, val_loader, optimizer, output_dir=fold_dir)
            trainer.start_training()

            self.validate(model, device)
            self.data_loader_factory.next_fold()

    def validate(self, model, device):
        """
        Validate the model for the current fold.

        Args:
            model: The model to validate.
            device: The device to use for validation.

        Returns:
            The validation result for the current fold.
        """
        print()
        print(f"Validating fold {self.data_loader_factory.current_fold + 1}/{self.data_loader_factory.k_folds}")
        val_loader = self.data_loader_factory.create_val_loader()
        result = Validation.load_and_validate(
            model=model,
            model_path=Config.LOGS_FOLDER + f'/fold{self.data_loader_factory.current_fold + 1}' + '/best_model.pth',
            val_loader=val_loader,
            device=device,
            output_dir=Config.LOGS_FOLDER + f'/fold{self.data_loader_factory.current_fold + 1}',
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
        print()
        print("\n" + "=" * 50)
        print()
        print(f"\t\tFinal Validation Using Test dataset")
        print("\n" + "=" * 50)
        test_loader = self.data_loader_factory.create_test_loader()
        test_dir = os.path.join(Config.LOGS_FOLDER, 'test_results')
        os.makedirs(test_dir, exist_ok=True)

        final_results = []
        for fold in range(self.data_loader_factory.k_folds):
            best_model_path = os.path.join(Config.LOGS_FOLDER, f'fold{fold + 1}', 'best_model.pth')
            model.load_state_dict(torch.load(best_model_path))
            fold_result = Validation.validate(
                model,
                test_loader,
                device,
                is_test_dataset=True,
                output_dir=os.path.join(test_dir, f'fold{fold + 1}')
            )
            final_results.append(fold_result)

        # Compute average results across folds
        avg_result = self.average_results(final_results)

        # Save average results
        self.save_final_results(avg_result, os.path.join(test_dir, 'average_results.json'))

        return final_results, avg_result

    @staticmethod
    def average_results(results):
        """
        Average the results across all folds.

        Args:
        results (list): A list of dictionaries, each containing the results for one fold.

        Returns:
        dict: A dictionary containing the averaged results across all folds.
        """
        if not results:
            return {}

        avg_result = {
            "multi_dices": [],
            "mean_multi_dice": [],
            "std_multi_dice": [],
            "confusion_matrix": None,
            "norm_confusion_matrix": None,
            "accuracy": 0
        }

        num_folds = len(results)
        num_classes = len(results[0]['mean_multi_dice'])

        # Average multi_dices
        all_multi_dices = [fold_result['multi_dices'] for fold_result in results]
        avg_result['multi_dices'] = np.mean(all_multi_dices, axis=0).tolist()

        # Average mean_multi_dice and std_multi_dice
        for i in range(num_classes):
            avg_result['mean_multi_dice'].append(
                float(np.mean([fold_result['mean_multi_dice'][i] for fold_result in results])))
            avg_result['std_multi_dice'].append(
                float(np.mean([fold_result['std_multi_dice'][i] for fold_result in results])))

        # Sum confusion matrices and then normalize
        sum_conf_matrix = np.sum([np.array(fold_result['confusion_matrix']) for fold_result in results], axis=0)
        avg_result['confusion_matrix'] = sum_conf_matrix.tolist()
        row_sums = sum_conf_matrix.sum(axis=1)
        avg_result['norm_confusion_matrix'] = (sum_conf_matrix / row_sums[:, np.newaxis]).tolist()

        # Average accuracy
        avg_result['accuracy'] = float(np.mean([fold_result['accuracy'] for fold_result in results]))

        # Calculate overall mean and std of Dice scores
        overall_mean_dice = float(np.mean(avg_result['mean_multi_dice']))
        overall_std_dice = float(np.mean(avg_result['std_multi_dice']))

        avg_result['overall_mean_dice'] = overall_mean_dice
        avg_result['overall_std_dice'] = overall_std_dice

        return avg_result

    @staticmethod
    def save_final_results(results: Dict[str, Union[float, List]], filepath: str) -> None:
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)

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
        self.train(model, device, optimizer)
        return self.fold_results, self.final_validation(model, device)
