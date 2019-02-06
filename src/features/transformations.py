from skimage import exposure
import numpy as np

from src.helper import get_config


config = get_config()

if config["preprocessing"]["use_augmentation"]:
    from imgaug import augmenters as iaa
    from imgaug.augmenters import Augmenter

def normalize_meanstd(x, axis=(1,2)): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.sqrt(((x - mean)**2).mean(axis=axis, keepdims=True))


    for i in range(x.shape[0]):
        mean = np.mean(x[i], axis=(0,1), keepdims=True)
        std = np.sqrt(((x[i] - mean)**2).mean(axis=(0,1), keepdims=True))

        if(np.any(std==0)):
            std[std==0]= 0.0000000001
        
        x[i] = (x[i] - mean) / std

    return x


def normalize_01(x, axis=None):
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)

    for i in range(x.shape[0]):
        x_min = x[i].min(axis=(0, 1), keepdims=True)
        x_max = x[i].max(axis=(0, 1), keepdims=True)

        if(np.any(x_max - x_max==0)):
            x_max[x_max==x_min]= x_max[x_max==x_min] + 0.0000000001

        
        x[i] = (x[i] - x_min)/(x_max-x_min)



    return x

def create_augmenter():

    hist_use = config["preprocessing"]["augmentation"]["histogram_method"]

    if hist_use == "rescale":
        hist_fun = RescaleIntensity
    elif hist_use == "CLAHE":
        hist_fun = iaa.CLAHE
    elif hist_use == "hist_equalization":
        hist_fun = iaa.AllChannelsHistogramEqualization
    else:
        hist_fun = None
    print("AAAA")

    list_augmenters = [
        iaa.SomeOf((0,5), [
            iaa.Affine(scale=(0.9, 1), mode="symmetric"),
            iaa.Affine(translate_percent=(0, 0.1), mode="symmetric"),
            iaa.Affine(rotate=(-15, 15), mode="symmetric"),
            iaa.Affine(shear=(-5, 5), mode="symmetric")
        ]),

        
        iaa.SomeOf((0,1),[
            iaa.GaussianBlur(sigma=(0, 0.5)), # blur images with a sigma of 0 to 0.5
            iaa.AverageBlur(k=(1, 3)),
            iaa.MedianBlur(k=1),

        ]),
        
        iaa.SomeOf( (0,2),[
        iaa.SaltAndPepper((0, 0.015), per_channel=True),
        iaa.OneOf([iaa.Multiply((0.9, 1.1)),
        iaa.Multiply((0.9, 1.1), per_channel=0.5)])
        ]
        )
    ]

    if hist_fun is not None:
        list_augmenters = list_augmenters + hist_fun(per_channel=False)
    
    seq = iaa.SomeOf((0, 5), list_augmenters)

    return seq

class EqualizeHist(Augmenter):
    def __init__(self,name=None, deterministic=False, random_state=None, per_channel = False):
        super(EqualizeHist, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        
        self.per_channel = per_channel

    def _augment_images(self, images, random_state, parents, hooks):
        for i in range(len(images)):
            if self.per_channel:
                tmp_image = images[i]
                
                for ch in arange(tmp_image.shape[2]):
                    tmp_image[: , :, ch] = exposure.equalize_hist(tmp_image[: , :, ch])
                    
                images[i] = tmp_image
            else:
                images[i] = exposure.equalize_hist(images[i])
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]

class RescaleIntensity(Augmenter):
    def __init__(self,name=None, deterministic=False, random_state=None, per_channel = False, percentiles = (2, 98)):
        super(RescaleIntensity, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        
        self.per_channel = per_channel
        self.percentiles = percentiles

    def _augment_images(self, images, random_state, parents, hooks):

        for i in range(len(images)):
            if self.per_channel:
                tmp_image = images[i]
                
                for ch in arange(tmp_image.shape[2]):
                    p1, p2 = np.percentile(img[: , : , ch], self.percentiles)
                    tmp_image[: ,: , ch] = exposure.rescale_intensity(tmp_image[: ,: , ch], in_range=(p1, p2))

                    
                images[i] = tmp_image
            else:
                p1, p2 = np.percentile(images[i], self.percentiles)
                images[i] = exposure.rescale_intensity(images[i], in_range=(p1, p2))
        return images

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import string
    alphabet = string.ascii_lowercase

    X = np.load("data/processed/X_tf_t2_kt.npy")
    y = np.load("data/processed/y_tf_t2_kt.npy")

    X = normalize_01(X)

    _augmenter = create_augmenter()

    fig = plt.figure()
    X_ = X[25]

    ax = fig.add_subplot(3,3,1)
    ax.imshow(X_, cmap="Greys_r")
    ax.set_title(alphabet[0])
    ax.axis('off')

    for i in range(2, 10):
        x_augmented = _augmenter.augment_image(X_)
        ax = fig.add_subplot(3,3,i)

        ax.imshow(x_augmented, cmap="Greys_r")
        ax.axis('off')
        ax.set_title(alphabet[i-1])

    plt.tight_layout()
    plt.show()
    #plt.savefig('/home/paulo/Projects/thesis/LaTeX/Chapters/Figures/chapter3/augmentations.pdf')


