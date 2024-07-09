from src.Config import Config
import numpy as np


def pad_to_size(image, target_shape=Config.PADDING_TARGET_SHAPE):
    current_shape = image.shape
    pad_width = [(0, max(target_shape[i] - current_shape[i], 0)) for i in range(3)]
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    return padded_image[:target_shape[0], :target_shape[1], :target_shape[2]]
