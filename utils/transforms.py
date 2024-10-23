import numpy as np


def compute_image_mean_and_std(images):
    """Calculating mean and standard deviation of each channel
    :param images (numpy arrays): images

    Returns:
    - mean (numpy array): mean of images per channel
    - std (numpy array): standard deviation of images per channel
    """
    mean, std = None, None

    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    return mean, std

class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    """

    def __init__(self, mean, std):
        """
        :param mean: mean of images per channel
        :param std: standard deviation of images per channel
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        images = (images - self.mean) / self.std
        return images


class RescaleTransform:
    """Transform class to rescale images to a given range"""

    def __init__(self, range_=(0, 1), old_range=(0, 255)):
        """
        :param range_: Value range to which images should be rescaled
        :param old_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = range_[0]
        self.max = range_[1]
        self._data_min = old_range[0]
        self._data_max = old_range[1]

    def __call__(self, images):
        images = images - self._data_min 
        images /= (self._data_max - self._data_min) 
        images *= (self.max - self.min) 
        images += self.min

        return images

class CombineTransforms:
    """Transform class that combines multiple other transforms into one"""

    def __init__(self, transforms):
        """
        :param transforms: transforms that can be applied to the image
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images