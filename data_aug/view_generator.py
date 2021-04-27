import numpy as np
from PIL import Image


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class FixImageChannels(object):
    def __call__(self, im):
        if im.mode != 'RGB':
            np_im = np.array(im)
            np_rgb_im = np.stack([np_im, np_im, np_im], axis=-1)
            return Image.fromarray(np_rgb_im)

        return im