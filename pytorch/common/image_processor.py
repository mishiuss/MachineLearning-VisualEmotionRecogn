import cv2
import numpy as np
import torch
from pytorch.common.image_preprocessing.cutout import Cutout
import torchvision.transforms as transforms


class TorchImageProcessor:
    """Simple data processors"""

    def __init__(self, image_size, is_color, mean, scale,
                 crop_size=0, pad=28, color='BGR',
                 use_cutout=False,
                 use_mirroring=False,
                 use_random_crop=False,
                 use_center_crop=False,
                 use_random_gray=False):
        """Everything that we need to init"""
        cv2.setNumThreads(1)
        torch.set_num_threads(1)

        self.image_size = image_size
        self.is_color = is_color

        if crop_size == 0:
            crop_size = image_size
        self.crop_size = crop_size

        self.color = color

        transforms_stack = [transforms.ToPILImage()]

        if not self.is_color:
            transforms_stack.append(transforms.Grayscale(num_output_channels=1))

        #transforms_stack.append(transforms.Resize(image_size))

        if use_mirroring:
            transforms_stack.append(transforms.RandomHorizontalFlip())

        if use_random_crop:
            transforms_stack.append(transforms.Pad(pad))
            transforms_stack.append(transforms.RandomCrop(crop_size))

        if use_center_crop:
            transforms_stack.append(transforms.Pad(pad))
            transforms_stack.append(transforms.CenterCrop(crop_size))

        transforms_stack.append(transforms.ToTensor())
        transforms_stack.append(transforms.Normalize((mean/255., mean/255., mean/255.), (1./(255.*scale), 1./(255.*scale), 1./(255.*scale))))

        self.transforms = transforms.Compose(transforms_stack)

        self.use_cutout = use_cutout
        self.cutout = Cutout(1, 56)

        self.use_random_gray = use_random_gray

    def crop_center(self, img, cropx, cropy):
        y, x, c = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx, :]

    def to_gray(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.stack([image, image, image], axis=2)

    def process(self, image_path):
        """
        Returns processed data.
        """
        try:
            image = cv2.imread(image_path)
        except:
            image = image_path

        if image is None:
            print(image_path)

        if self.is_color and image.shape[2] == 1:
            image = np.stack([image, image, image], axis=2)

        if image.shape[0] != self.image_size[0] or image.shape[1] != self.image_size[1]:
            image = cv2.resize(image, dsize=(self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_CUBIC)

        if self.color == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.color == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #if 0:
        #    cimg = self.crop_center(image, self.crop_size, self.crop_size)
        #    cv2.imwrite('temp.png', cimg)

        if self.use_random_gray:
            if np.random.uniform(0, 1) < 0.05:
                image = self.to_gray(image)

        if self.use_cutout:
            image = self.cutout.cut(image)

        image = self.transforms(image).numpy()

        return image