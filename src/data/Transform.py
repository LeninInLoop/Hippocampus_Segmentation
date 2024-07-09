
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class ZNormalization:
    def __call__(self, image):
        return (image - image.mean()) / image.std()
