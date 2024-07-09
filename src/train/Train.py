from src.train import *


class Train:
    def __init__(self, model, device, train_loader, val_loader, optimizer, criterion, scheduler, config,
                 logs_folder=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.logs_folder = logs_folder

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for i, (batch_images, batch_labels) in enumerate(self.train_loader):
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_images)
            loss = self.criterion(outputs, batch_labels, device=self.device)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:  # Print every 10 batches
                print(f'    Batch {i + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_images, batch_labels in self.val_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_labels, device=self.device)

                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def start_training(self):
        best_val_loss = float('inf')
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()

            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Time: {epoch_elapsed_time:.1f} seconds")

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.logs_folder is not None:
                    checkpoint_path = os.path.join(self.logs_folder, 'best_model.pth')
                    torch.save(self.model.state_dict(), checkpoint_path)

            # Save checkpoint every 'val_epochs'
            if (epoch + 1) % self.config.val_epochs == 0 and self.logs_folder is not None:
                checkpoint_path = os.path.join(self.logs_folder, f'model_epoch_{epoch + 1:04d}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)

        print('Training ended!')