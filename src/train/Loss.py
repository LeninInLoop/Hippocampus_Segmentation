from typing import Union, List
from src.utils import *


def dice_loss_coefficient(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Dice coefficient between outputs and labels.

    Args:
        outputs (torch.Tensor): Predicted outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: Dice loss (1 - Dice coefficient).
    """
    eps = 1e-5
    outputs, labels = outputs.float().flatten(), labels.float().flatten()
    intersect = torch.dot(outputs, labels)
    union = torch.sum(outputs) + torch.sum(labels)
    dice = (2 * intersect + eps) / (union + eps)
    return 1 - dice


def dice_coefficient(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Dice coefficient between outputs and labels.

    Args:
        outputs (torch.Tensor): Predicted outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: Dice loss (1 - Dice coefficient).
    """
    eps = 1e-5
    outputs, labels = outputs.float().flatten(), labels.float().flatten()
    intersect = torch.dot(outputs, labels)
    union = torch.sum(outputs) + torch.sum(labels)
    dice = (2 * intersect + eps) / (union + eps)
    return dice


def one_hot_encode(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Perform one-hot encoding on the input label tensor.

    Args:
        label (torch.Tensor): Input label tensor of shape BxHxW or BxDxHxW.
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: One-hot encoded tensor.
    """
    assert 3 <= len(label.shape) <= 4, f'Invalid Label Shape {label.shape}'

    shape = (label.shape[0], num_classes) + label.shape[1:]
    label_ohe = torch.zeros(shape)

    for batch_idx, batch_el_label in enumerate(label):
        for cls in range(num_classes):
            label_ohe[batch_idx, cls] = (batch_el_label == cls)

    return label_ohe.long()


def multi_class_dice_loss(outputs: torch.Tensor, labels: torch.Tensor, do_one_hot: bool = False,
                          get_list: bool = False, device: torch.device = None) -> Union[
    List[torch.Tensor], torch.Tensor]:
    """
    Compute the multi-class Dice coefficient.

    Args:
        outputs (torch.Tensor): CNN output probabilities of shape [BxKxHxW].
        labels (torch.Tensor): Ground truth of shape [BxKxHxW] or [BxHxW].
        do_one_hot (bool): Set to True if ground truth has shape [BxHxW].
        get_list (bool): Set to True to return a list of Dice coefficients per class.
        device (torch.device): CUDA device for computation.

    Returns:
        Union[List[torch.Tensor], torch.Tensor]: List of Dice coefficients or average Dice coefficient.
    """
    num_classes = outputs.shape[1]
    if do_one_hot:
        labels = one_hot_encode(labels, num_classes).to(device)

    dices = [dice_loss_coefficient(outputs[:, cls].unsqueeze(1), labels[:, cls].unsqueeze(1))
             for cls in range(1, num_classes)]

    return dices if get_list else sum(dices) / (num_classes - 1)


def multi_class_dice(outputs: torch.Tensor, labels: torch.Tensor, do_one_hot: bool = False,
                     get_list: bool = True, device: torch.device = None) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Compute the multi-class Dice coefficient.

    Args:
        outputs (torch.Tensor): CNN output probabilities of shape [BxKxHxW].
        labels (torch.Tensor): Ground truth of shape [BxKxHxW] or [BxHxW].
        do_one_hot (bool): Set to True if ground truth has shape [BxHxW].
        get_list (bool): Set to True to return a list of Dice coefficients per class.
        device (torch.device): CUDA device for computation.

    Returns:
        Union[List[torch.Tensor], torch.Tensor]: List of Dice coefficients or average Dice coefficient.
    """
    num_classes = outputs.shape[1]
    if do_one_hot:
        labels = one_hot_encode(labels, num_classes).to(device)

    dices = [dice_coefficient(outputs[:, cls].unsqueeze(1), labels[:, cls].unsqueeze(1))
             for cls in range(1, num_classes)]

    return dices if get_list else sum(dices) / (num_classes - 1)


def get_multi_dice_loss(outputs: torch.Tensor, labels: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    """
    Compute the multi-class Dice loss.

    Args:
        outputs (torch.Tensor): CNN output probabilities.
        labels (torch.Tensor): Ground truth labels.
        device (torch.device): CUDA device for computation.

    Returns:
        torch.Tensor: Multi-class Dice loss.
    """
    return multi_class_dice_loss(outputs, labels[:, 0], do_one_hot=True, get_list=False, device=device)
