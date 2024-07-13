from src.train import *


class Train:
    """
    A class to handle the training process of a deep learning model.
    """

    def __init__(self, model, device, train_loader, val_loader, optimizer):
        """
        Initialize the Train class with model and training parameters.

        Args:
            model: The neural network model to be trained.
            device: The device (CPU/GPU) to run the training on.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            optimizer: The optimizer for updating model parameters.
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self._create_logs_folder()

    @staticmethod
    def _create_logs_folder():
        """Create a folder for storing logs and checkpoints if it doesn't exist."""
        if not os.path.isdir(Config.LOGS_FOLDER):
            os.makedirs(Config.LOGS_FOLDER, exist_ok=True)

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            float: Average loss for this epoch.
        """
        self.model.train()
        total_loss = 0
        for i, (batch_images, batch_labels) in enumerate(self.train_loader):
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_images)
            loss = get_multi_dice_loss(outputs, batch_labels, device=self.device)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            print(f'    Batch {i + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        return total_loss / len(self.train_loader)

    def validate(self):
        """
        Validate the model on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_images, batch_labels in self.val_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_images)
                loss = get_multi_dice_loss(outputs, batch_labels, device=self.device)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def start_training(self):
        """
        Start the training process for the specified number of epochs.
        """
        best_val_loss = float('inf')
        for epoch in range(Config.NUM_EPOCHS):
            epoch_start_time = time.time()

            print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
            train_loss = self.train_epoch()
            val_loss = self.validate()

            epoch_elapsed_time = time.time() - epoch_start_time

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Time: {epoch_elapsed_time:.1f} seconds")

            self._save_model(epoch, val_loss, best_val_loss)
            best_val_loss = min(best_val_loss, val_loss)

        print('Training ended!')

    def _save_model(self, epoch, val_loss, best_val_loss):
        """
        Save the model if it's the best so far, and periodically.

        Args:
            epoch (int): Current epoch number.
            val_loss (float): Current validation loss.
            best_val_loss (float): Best validation loss so far.
        """
        if val_loss < best_val_loss:
            self._save_checkpoint('best_model.pth')
            print(f"New best model saved with validation loss: {val_loss:.4f}")

        if (epoch + 1) % Config.VAL_EPOCHS == 0:
            self._save_checkpoint(f'model_epoch_{epoch + 1:04d}.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}")

    def _save_checkpoint(self, filename):
        """
        Save a checkpoint of the model.

        Args:
            filename (str): Name of the file to save the checkpoint.
        """
        if Config.LOGS_FOLDER is not None:
            checkpoint_path = os.path.join(Config.LOGS_FOLDER, filename)
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
