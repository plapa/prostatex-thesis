from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter
from skimage import exposure
import numpy as np

from src.helper import get_config

def normalize_meanstd(x, axis=(1,2)): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.sqrt(((x - mean)**2).mean(axis=axis, keepdims=True))
    return (x - mean) / std


def normalize_01(x, axis=None):
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)

    return (x - x_min)/(x_max-x_min)

def create_augmenter():
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 8)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        
        iaa.SomeOf((0,3), [
            iaa.Fliplr(1),
            iaa.Flipud(1),
        ]),
        
        iaa.SomeOf((0,1),[
            iaa.GaussianBlur(sigma=(0, 0.5)), # blur images with a sigma of 0 to 3.0
            iaa.AverageBlur(k=(1, 3)),
            iaa.MedianBlur(k=1),

        ]),
        
        iaa.SomeOf((0,2), [
        iaa.SaltAndPepper(0.01, per_channel=True),
        iaa.Dropout(p=0.01, per_channel=True),
        iaa.OneOf([iaa.Multiply((0.5, 1.5)),
        iaa.Multiply((0.5, 1.5), per_channel=0.5)])
        
        ])
    ])

    return seq

class Equalize_Hist(Augmenter):
    def __init__(self,name=None, deterministic=False, random_state=None, per_channel = False):
        super(Equalize_Hist, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        
        self.per_channel = False

    def _augment_images(self, images, random_state, parents, hooks):
        for i in range(len(images)):
                images[i] = exposure.equalize_hist(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]
