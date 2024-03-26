import os
import random

import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class FootUlcerSegmentationDataset(Dataset):
    """
    Loads dataset from the Foot Ulcer Segmentation Challenge.

    Parameters
    ----------
    img_dir : `str`
        Path to the directory containing the images.
    seg_dir : `str``
        Path to the directory containing the segmentation masks.
    mandatoryTransforms : `T.Compose | None`
        Transformations that will always be applied to the dataset. Typically, this is
        set to `T.Compose([ToTensor()])` which turns the images and segmentation masks
        into tensors pytorch can work with.
    transform : `T.Compose | None`
        Transformations to be applied to the dataset.
    aug_percentage : `float`
        Sets the percentage of images to be augmentated. Set between `0.0` and `1.0`.
    """

    def __get_all_files(self, path: os.PathLike) -> list:
        """Returns all files from specified directory."""
        filelist = os.listdir(path)
        filepaths = []
        for f in filelist:
            filepaths.append(os.path.join(path, f))
        return filepaths

    def __init__(
        self,
        img_dir,
        seg_dir,
        mandatoryTransform=None,
        transform=None,
        aug_percentage=1.0,
    ):

        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.mandatoryTransform = mandatoryTransform
        self.transform = transform
        self.aug_percentage = aug_percentage

        images = self.__get_all_files(img_dir)
        segmentations = self.__get_all_files(seg_dir)
        self.data = pd.DataFrame(
            data={"images": images, "segmentations": segmentations}
        )

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        image = io.imread(self.data.iloc[idx, 0])
        segmentation = io.imread(self.data.iloc[idx, 1])

        if random.random() <= self.aug_percentage and self.transform:
            image, segmentation = self.transform((image, segmentation))

        if self.mandatoryTransform is not None:
            image, segmentation = self.mandatoryTransform((image, segmentation))

        return image, segmentation
