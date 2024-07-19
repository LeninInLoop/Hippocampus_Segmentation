from src.train.Loss import multi_class_dice
from src.utils import *


class Validation:
    @classmethod
    def validate(cls, model, val_loader, device, is_test_dataset=False):
        print("Starting Validation...")
        model.eval()
        multi_dices = []
        all_labels = []
        all_predictions = []
        output_dir = Config.TEST_LOGS_FOLDER if is_test_dataset else Config.LOGS_FOLDER

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Validation")):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs2 = torch.argmax(outputs, dim=1)  # B x Z x Y x X

                multi_dice = multi_class_dice(outputs, labels[:, 0], do_one_hot=True, get_list=True, device=device)
                multi_dices.append([dice.cpu().numpy() for dice in multi_dice])  # Move to CPU and convert to numpy

                # Flatten labels and predictions for confusion matrix and accuracy
                all_labels.extend(labels.cpu().numpy().flatten())
                all_predictions.extend(outputs2.cpu().numpy().flatten())

                # Save example images
                if batch_idx == 0:
                    cls.save_example_images(inputs, labels, outputs2, output_dir)

        print(f"Example 3D visualizations saved in {Config.LOGS_FOLDER}")
        multi_dices_np = np.array(multi_dices)
        mean_multi_dice = np.mean(multi_dices_np, axis=0)
        std_multi_dice = np.std(multi_dices_np, axis=0)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions, labels=range(3))

        # Calculate normalized confusion matrix
        norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)

        results = {
            "multi_dices": multi_dices,
            "mean_multi_dice": mean_multi_dice,
            "std_multi_dice": std_multi_dice,
            "confusion_matrix": conf_matrix,
            "norm_confusion_matrix": norm_conf_matrix,
            "accuracy": accuracy,
        }

        cls.print_results(results)
        cls.save_confusion_matrix(conf_matrix, norm_conf_matrix, 3, output_dir)
        return results

    @staticmethod
    def print_results(results):
        """
        Print validation results.

        Args:
            results (dict): Dictionary containing validation metrics.
        """
        print("\nValidation Results:")
        for i, (mean, std) in enumerate(zip(results['mean_multi_dice'], results['std_multi_dice'])):
            print(f"Class {i + 1} Dice: {mean:.4f} +/- {std:.4f}")
        print(
            f"Mean Multi-Dice: {np.mean(results['mean_multi_dice'][1:]):.4f} +/- {np.mean(results['std_multi_dice'][1:]):.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")

    @staticmethod
    def save_example_images(inputs, labels, predictions, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for batch_idx in range(inputs.shape[0]):
            input_vol = inputs[batch_idx, 0].cpu().numpy()
            label_vol = labels[batch_idx, 0].cpu().numpy()
            pred_vol = predictions[batch_idx].cpu().numpy()

            # # Print volume statistics
            # Validation.print_volume_stats(input_vol, "Input")
            # Validation.print_volume_stats(label_vol, "Label")
            # Validation.print_volume_stats(pred_vol, "Prediction")

            # Create subplots
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=('Input', 'Ground Truth', 'Prediction')
            )

            def add_voxel_plot(fig, vol, row, col, colorscale):
                # Ensure the volume has valid shapes and ranges
                x, y, z = np.meshgrid(
                    np.arange(vol.shape[0]),
                    np.arange(vol.shape[1]),
                    np.arange(vol.shape[2]),
                    indexing='ij'
                )
                vol_norm = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))  # Normalize volume

                fig.add_trace(
                    go.Volume(
                        x=x.flatten(),
                        y=y.flatten(),
                        z=z.flatten(),
                        value=vol_norm.flatten(),
                        opacity=0.5,  # Adjust this value to change the transparency
                        surface_count=50,  # Adjust this value to change the smoothness
                        colorscale=colorscale,
                    ),
                    row=row, col=col
                )

                return fig

            # Add voxel plots
            add_voxel_plot(fig, input_vol, 1, 1, 'Gray')
            add_voxel_plot(fig, label_vol, 1, 2, 'Viridis')
            add_voxel_plot(fig, pred_vol, 1, 3, 'Inferno')

            # Update layout
            fig.update_layout(
                title_text=f"3D Visualization - Batch {batch_idx}",
                width=1800,
                height=600
            )

            # Update scenes
            for i in range(1, 4):
                fig.update_scenes(
                    aspectmode='data',
                    xaxis_visible=True,
                    yaxis_visible=True,
                    zaxis_visible=True,
                    row=1, col=i
                )

            # Save as interactive HTML
            output_file = os.path.join(output_dir, f"example_3d_{batch_idx}.html")
            fig.write_html(output_file)

    @staticmethod
    def print_volume_stats(vol, name):
        print(f"{name} stats:")
        print(f"  Shape: {vol.shape}")
        print(f"  Min: {np.min(vol)}")
        print(f"  Max: {np.max(vol)}")
        print(f"  Mean: {np.mean(vol)}")
        print(f"  Std: {np.std(vol)}")
        print(f"  Unique values: {np.unique(vol)}")

    @staticmethod
    def save_confusion_matrix(cm, norm_cm, num_classes, output_dir=Config.LOGS_FOLDER):
        """
        Save confusion matrices as CSV files and a combined image.

        Args:
        cm (numpy.ndarray): The confusion matrix.
        norm_cm (numpy.ndarray): The normalized confusion matrix.
        num_classes (int): The number of classes in the classification problem.
        output_dir (str): The directory to save the outputs. Defaults to Config.LOGS_FOLDER.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save raw confusion matrix as CSV
        df_cm = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))
        csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
        df_cm.to_csv(csv_path)
        print(f"Confusion matrix saved as CSV: {csv_path}")

        # Save normalized confusion matrix as CSV
        df_norm_cm = pd.DataFrame(norm_cm, index=range(num_classes), columns=range(num_classes))
        norm_csv_path = os.path.join(output_dir, 'normalized_confusion_matrix.csv')
        df_norm_cm.to_csv(norm_csv_path)
        print(f"Normalized confusion matrix saved as CSV: {norm_csv_path}")

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot raw confusion matrix
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # Plot normalized confusion matrix
        sns.heatmap(df_norm_cm, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
        ax2.set_title('Normalized Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')

        # Adjust layout and save the figure
        plt.tight_layout()
        img_path = os.path.join(output_dir, 'confusion_matrices.png')
        plt.savefig(img_path)
        plt.close()  # Close the figure to free up memory
        print(f"Confusion matrices saved as image: {img_path}")

    @classmethod
    def load_and_validate(cls, model, model_path, val_loader, device, is_test_dataset=False):
        """
        Load a model from a checkpoint and perform validation.

        Args:
            model (nn.Module): The model architecture.
            model_path (str): Path to the model checkpoint.
            val_loader (DataLoader): DataLoader for the validation dataset.
            device (torch.device): The device to run the validation on.
            is_test_dataset (bool): Whether the dataset is testing dataset.

        Returns:
            dict: A dictionary containing validation metrics.
        """
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return cls.validate(model, val_loader, device, is_test_dataset)
