from src.train import *


class Train:
    def __init__(self, model, device, train_loader, val_loader, optimizer, criterion, scheduler):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.best_val_dice = 0

    def start_training(self):
        for epoch in range(1, Config.NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
            print("-" * 20)

            self.train_epoch(epoch)
            val_loss, val_dice = self.validate()

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved with Dice score: {self.best_val_dice:.4f}")

        print(f'Training completed. Best validation Dice score: {self.best_val_dice:.4f}')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_y_true = []
        all_y_pred = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            all_y_true.extend(target.cpu().numpy().flatten())
            all_y_pred.extend(predicted.cpu().numpy().flatten())

            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        metrics = self.calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
        self.print_metrics("Training", avg_loss, metrics)

    def validate(self):
        self.model.eval()
        val_loss = 0
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                _, predicted = torch.max(output.data, 1)
                all_y_true.extend(target.cpu().numpy().flatten())
                all_y_pred.extend(predicted.cpu().numpy().flatten())

        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        metrics = self.calculate_metrics(np.array(all_y_true), np.array(all_y_pred))
        self.print_metrics("Validation", val_loss, metrics)

        return val_loss, metrics['dice']

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        return {
            'cm': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'dice': dice,
            'iou': iou
        }

    @staticmethod
    def print_metrics(phase, loss, metrics):
        print(f'\n{phase} set: Average loss: {loss:.4f}')
        print(f'Confusion Matrix:\n{metrics["cm"]}')
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'Precision: {metrics["precision"]:.4f}')
        print(f'Recall: {metrics["recall"]:.4f}')
        print(f'Dice Coefficient: {metrics["dice"]:.4f}')
        print(f'IoU: {metrics["iou"]:.4f}\n')