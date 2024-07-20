from abc import ABC, abstractmethod
from .Train import *
from ..data import DataSetLoader


class TrainingStrategy(ABC):
    def __init__(self, data_loader_factory):
        self.data_loader_factory = data_loader_factory

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def train(self, model, device, optimizer):
        pass

    @abstractmethod
    def validate(self, model, device):
        pass

    @abstractmethod
    def final_validation(self, model, device):
        pass

    def execute(self, model, device, optimizer):
        self.prepare_data()
        self.train(model, device, optimizer)
        val_result = self.validate(model, device)
        final_results = self.final_validation(model, device)
        return val_result, final_results


class StandardTrainingStrategy(TrainingStrategy):
    def __init__(self, data_loader_factory):
        super().__init__(data_loader_factory)
        dataset = DataSetLoader(data_loader_factory)
        self.train_loader = dataset.train_loader
        self.val_loader = dataset.val_loader
        self.test_loader = dataset.test_loader

    def prepare_data(self):
        # No additional preparation needed
        pass

    def train(self, model, device, optimizer):
        trainer = Train(model, device, self.train_loader, self.val_loader, optimizer)
        trainer.start_training()

    def validate(self, model, device):
        print("Validating Using validation Dataset:")
        return Validation.load_and_validate(
            model=model,
            model_path=Config.BEST_MODEL_SAVE_PATH,
            val_loader=self.val_loader,
            device=device
        )

    def final_validation(self, model, device):
        print("Validating Using Test Dataset:")
        return Validation.load_and_validate(
            model=model,
            model_path=Config.BEST_MODEL_SAVE_PATH,
            val_loader=self.test_loader,
            device=device,
            is_test_dataset=True
        )


class KFoldTrainingStrategy(TrainingStrategy):
    def __init__(self, data_loader_factory):
        super().__init__(data_loader_factory)
        self.fold_results = []

    def prepare_data(self):
        self.data_loader_factory.prepare_folds()

    def train(self, model, device, optimizer):
        train_loader = self.data_loader_factory.create_train_loader()
        val_loader = self.data_loader_factory.create_val_loader()
        trainer = Train(model, device, train_loader, val_loader, optimizer)
        trainer.start_training()

    def validate(self, model, device):
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
        print("Final validation using average of fold validations:")
        return sum(self.fold_results) / len(self.fold_results)

    def execute(self, model, device, optimizer):
        self.prepare_data()
        for _ in range(self.data_loader_factory.k_folds):
            self.train(model, device, optimizer)
            self.validate(model, device)
            self.data_loader_factory.next_fold()
        return self.fold_results, self.final_validation(model, device)
