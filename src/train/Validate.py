from src.utils import *


class Validation:
    @classmethod
    def validate(cls, model, val_loader, device):
        print("Starting Validation...")
        model.eval()
        multi_dices = []
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                inputs, labels = data['t1']['data'], data['label']['data']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs = torch.argmax(outputs, dim=1)  # B x Z x Y x X

                outputs_np = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()[:, 0]  # B x Z x Y x X

                multi_dice = cls.multi_dice_coeff(labels_np, outputs_np, 3)
                multi_dices.append(multi_dice)

                # Flatten labels and predictions for confusion matrix
                all_labels.extend(labels_np.flatten())
                all_predictions.extend(outputs_np.flatten())

        multi_dices_np = np.array(multi_dices)
        mean_multi_dice = np.mean(multi_dices_np)
        std_multi_dice = np.std(multi_dices_np)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(3))

        results = {
            "multi_dices": multi_dices,
            "mean_multi_dice": mean_multi_dice,
            "std_multi_dice": std_multi_dice,
            "confusion_matrix": conf_matrix
        }

        cls.print_results(results)
        cls.save_confusion_matrix(conf_matrix, 3, Config.LOGS_FOLDER)
        return results

    @staticmethod
    def multi_dice_coeff(y_true, y_pred, num_classes):
        """
        Calculate multi-class Dice coefficient.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.
            num_classes (int): Number of classes.

        Returns:
            float: Multi-class Dice coefficient.
        """
        dice_scores = []
        for class_idx in range(num_classes):
            intersection = np.sum((y_true == class_idx) & (y_pred == class_idx))
            union = np.sum(y_true == class_idx) + np.sum(y_pred == class_idx)
            dice = (2. * intersection + 1e-5) / (union + 1e-5)
            dice_scores.append(dice)
        return np.mean(dice_scores)

    @staticmethod
    def print_results(results):
        """
        Print validation results.

        Args:
            results (dict): Dictionary containing validation metrics.
        """
        print("\nValidation Results:")
        print(f"Mean Multi-Dice: {results['mean_multi_dice']:.4f} +/- {results['std_multi_dice']:.4f}")

    @staticmethod
    def save_confusion_matrix(cm, num_classes, output_dir=Config.LOGS_FOLDER):
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save as CSV
        df_cm = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))
        csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
        df_cm.to_csv(csv_path)
        print(f"Confusion matrix saved as CSV: {csv_path}")

        # Save as image
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        img_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(img_path)
        plt.close()
        print(f"Confusion matrix saved as image: {img_path}")

    @classmethod
    def load_and_validate(cls, model, model_path, val_loader, device):
        """
        Load a model from a checkpoint and perform validation.

        Args:
            model (nn.Module): The model architecture.
            model_path (str): Path to the model checkpoint.
            val_loader (DataLoader): DataLoader for the validation dataset.
            device (torch.device): The device to run the validation on.
            config: Configuration object containing settings.

        Returns:
            dict: A dictionary containing validation metrics.
        """
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return cls.validate(model, val_loader, device)
