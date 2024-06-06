import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy


def generate_random_data(height, width, count):
    x, y = zip(*[generate_img_and_mask(height, width) for _ in range(count)])
    X = np.asarray(x) * 255
    X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    Y = np.asarray(y)
    return X, Y


def generate_img_and_mask(height, width):
    shape = (height, width)
    locations = [get_random_location(*shape, zoom=z) for z in [1.0, 0.7, 0.5, 1.0, 0.8, 1.2]]

    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *locations[0])
    arr = add_circle(arr, *locations[1])
    arr = add_circle(arr, *locations[2], fill=True)
    arr = add_mesh_square(arr, *locations[3])
    arr = add_filled_square(arr, *locations[4])
    arr = add_plus(arr, *locations[5])
    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    masks = np.asarray([
        add_filled_square(np.zeros(shape, dtype=bool), *locations[4]),
        add_circle(np.zeros(shape, dtype=bool), *locations[2], fill=True),
        add_triangle(np.zeros(shape, dtype=bool), *locations[0]),
        add_circle(np.zeros(shape, dtype=bool), *locations[1]),
        add_filled_square(np.zeros(shape, dtype=bool), *locations[3]),
        add_plus(np.zeros(shape, dtype=bool), *locations[5])
    ]).astype(np.float32)

    return arr, masks


def add_shape(arr, x, y, size, shape_fn):
    s = int(size / 2)
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    return np.logical_or(arr, shape_fn(xx, yy, x, y, s))


def add_filled_square(arr, x, y, size):
    return add_shape(arr, x, y, size, lambda xx, yy, x, y, s: (xx > x - s) & (xx < x + s) & (yy > y - s) & (yy < y + s))


def add_mesh_square(arr, x, y, size):
    return add_shape(arr, x, y, size, lambda xx, yy, x, y, s: (xx > x - s) & (xx < x + s) & (xx % 2 == 1) & (yy > y - s) & (yy < y + s) & (yy % 2 == 1))


def add_triangle(arr, x, y, size):
    s = int(size / 2)
    triangle = np.tril(np.ones((size, size), dtype=bool))
    arr[x - s:x - s + triangle.shape[0], y - s:y - s + triangle.shape[1]] = triangle
    return arr


def add_circle(arr, x, y, size, fill=False):
    return add_shape(arr, x, y, size, lambda xx, yy, x, y, s: (np.sqrt((xx - x) ** 2 + (yy - y) ** 2) < size) & (fill | (np.sqrt((xx - x) ** 2 + (yy - y) ** 2) >= size * 0.7)))


def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x - 1:x + 1, y - s:y + s] = True
    arr[x - s:x + s, y - 1:y + 1] = True
    return arr


def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))
    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)
    return x, y, size


def plot_side_by_side(img_arrays):
    flatten_list = [item for sublist in zip(*img_arrays) for item in sublist]
    plot_img_array(np.array(flatten_list))


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img_array[i])
    plt.show()


def masks_to_colorimg(masks):
    colors = np.array([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228), (56, 34, 132), (160, 194, 56)])
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    for y in range(masks.shape[1]):
        for x in range(masks.shape[2]):
            selected_colors = colors[masks[:, y, x] > 0.5]
            if len(selected_colors) > 0:
                colorimg[y, x, :] = np.mean(selected_colors, axis=0)
    return colorimg.astype(np.uint8)


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = generate_random_data(192, 192, count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)
        return image, mask


def get_data_loaders(batch_size=25):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = SimDataset(100, transform)
    val_set = SimDataset(20, transform)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=25):
    dataloaders = get_data_loaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and epoch_loss < best_loss:
                print("Saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    print(f'Best val loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp


def run(UNet):
    num_class = 6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=5)
    model.eval()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = SimDataset(3, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        pred = model(inputs)
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()

    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]
    target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    pred_rgb = [masks_to_colorimg(x) for x in pred]

    plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

