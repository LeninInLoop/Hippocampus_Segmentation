from src.utils import *
import torchio as tio
from torchio import ZNormalization, Compose, TYPE, DATA, Subject, RandomAffine, RandomElasticDeformation, \
    RandomNoise, RandomBlur
from torchio.transforms import Lambda
from src.data.Helper import get_pad_3d_image


class LambdaChannel(Lambda):
    """
    A custom Lambda transform that applies a function to each channel of an image.
    """

    def apply_transform(self, sample: Subject) -> dict:
        for image in sample.get_images(intensity_only=False):
            if self.types_to_apply and image[TYPE] not in self.types_to_apply:
                continue

            result = self.function(image[DATA])
            self._validate_result(result, image[DATA])
            image[DATA] = result
        return sample

    @staticmethod
    def _validate_result(result, original):
        if not isinstance(result, torch.Tensor):
            raise ValueError(f"Result must be a torch.Tensor, not {type(result)}")
        if result.dtype != torch.float32:
            raise ValueError(f"Result must have dtype torch.float32, not {result.dtype}")
        if result.ndim != original.ndim:
            raise ValueError(f"Result must have {original.ndim} dimensions, not {result.ndim}")


class CustomCompose(Compose):
    """
    A custom Compose transform that allows for probabilistic application of transforms.
    """

    def __init__(self, transforms_dict):
        transforms = [
            tio.OneOf({transform: prob}) if isinstance(transform, tio.Transform) else transform
            for transform, prob in transforms_dict.items()
        ]
        super().__init__(transforms)

    def __call__(self, sample):
        subject = Subject(
            image=tio.ScalarImage(tensor=sample['image']),
            label=tio.LabelMap(tensor=sample['label'])
        )
        transformed = super().__call__(subject)
        return {
            'image': transformed['image'].data,
            'label': transformed['label'].data
        }


# Define the transforms
train_transforms_dict = {
    ZNormalization(): 1.0,
    RandomAffine(): 0.05,
    RandomElasticDeformation(max_displacement=3): 0.20,
    RandomNoise(std=(0, 0.1)): 0.10,
    RandomBlur(std=(0, 0.1)): 0.10,
    LambdaChannel(get_pad_3d_image(pad_ref=Config.PADDING_TARGET_SHAPE, zero_pad=False)): 1.0,
}

validation_transforms_dict = {
    ZNormalization(): 1.0,
    LambdaChannel(get_pad_3d_image(pad_ref=Config.PADDING_TARGET_SHAPE, zero_pad=False)): 1.0,
}

train_transform = CustomCompose(train_transforms_dict)
validation_transform = CustomCompose(validation_transforms_dict)
