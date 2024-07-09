import torch


def dice(outputs, labels):
    eps = 1e-5
    outputs, labels = outputs.float(), labels.float()
    # Flatten all dimensions except the batch dimension
    outputs = outputs.view(outputs.size(0), -1)
    labels = labels.view(labels.size(0), -1)
    intersect = (outputs * labels).sum(dim=1)
    union = outputs.sum(dim=1) + labels.sum(dim=1)
    dice_coeff = (2 * intersect + eps) / (union + eps)
    dice_loss = 1 - dice_coeff.mean()
    return dice_loss


def one_hot_encode(label, num_classes):
    """ Torch One Hot Encode
    :param label: Tensor of shape BxHxW or BxDxHxW
    :param num_classes: K classes
    :return: label_ohe, Tensor of shape BxKxHxW or BxKxDxHxW
    """
    assert len(label.shape) == 3 or len(label.shape) == 4, 'Invalid Label Shape {}'.format(label.shape)
    label_ohe = None
    if len(label.shape) == 3:
        label_ohe = torch.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]))
    elif len(label.shape) == 4:
        label_ohe = torch.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2], label.shape[3]))
    for batch_idx, batch_el_label in enumerate(label):
        for cls in range(num_classes):
            label_ohe[batch_idx, cls] = (batch_el_label == cls)
    label_ohe = label_ohe.long()
    return label_ohe


def dice_n_classes(outputs, labels, do_one_hot=False, get_list=False, device=None):
    num_classes = outputs.shape[1]
    if do_one_hot:
        labels = one_hot_encode(labels, num_classes)
        labels = labels.cuda(device=device)

    dices = list()
    for cls in range(1, num_classes):
        outputs_ = outputs[:, cls].unsqueeze(dim=1)
        labels_ = labels[:, cls].unsqueeze(dim=1)
        dice_ = dice(outputs_, labels_)
        dices.append(dice_)
    if get_list:
        return dices
    else:
        return sum(dices) / (num_classes - 1)


def get_multi_dice_loss():
    def loss_fn(outputs, labels, device=None):
        return dice_n_classes(outputs, labels, do_one_hot=True, get_list=False, device=device)
    return loss_fn
