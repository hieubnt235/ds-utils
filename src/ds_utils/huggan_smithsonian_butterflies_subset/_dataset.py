from collections.abc import Iterable
from math import ceil
from typing import TypedDict, Sequence, Callable

import torch
from datasets import load_dataset
from torch.utils.data.dataset import _T_co
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset as ArrDataset


class SampleType(TypedDict):
    raw_images: list[Image.Image]
    images: torch.Tensor
    """
    Float tensor with shape (batchsize, channels, height, width), values must in range [-1,1]
    """


def check_sample(samples: SampleType):
    assert (imgs := samples["images"]).max() <= 1.0
    assert imgs.min() >= -1.0
    assert torch.is_floating_point(imgs)
    assert len((s := imgs.shape)) == 4
    assert s[1] == 3


class DefaultTransformKwargs(TypedDict):
    resize: int | Sequence[int]
    p_h_flip: float


class HugganSmithsonianButterfliesSubsetDataset(Dataset[SampleType]):
    ds_path = "huggan/smithsonian_butterflies_subset"
    df_tf_kwargs = DefaultTransformKwargs(resize=(128, 128), p_h_flip=0.5)

    def __init__(
        self,
        transform: Callable[[list[Image.Image]], torch.Tensor] | None = None,
        **default_tf_kwargs,
    ):
        """

        Args:
            transform: Fuction with input is list[Image] and return Float tensor with shape (batchsize, channels, height, width),
             values must in range [-1,1].
            **default_tf_kwargs:
        """

        self._dataset: ArrDataset = load_dataset(self.ds_path, split="train")
        self.tf_func: Callable[[list[Image.Image]], SampleType] | None = None
        self._to_resized_images: v2.Transform | None = None
        self._transform: v2.Transform | None = None

        if transform:
            self.tf_func: Callable[[list[Image.Image]], SampleType] = (
                lambda imgs: SampleType(raw_images=imgs, images=transform(imgs))
            )
        else:
            tf_kw = self.df_tf_kwargs
            tf_kw.update(default_tf_kwargs)

            self._to_resized_images = v2.Compose(
                [
                    v2.ToImage(),
                    # Always resize in the very first of pipeline, when data is in uint8
                    v2.Resize(tf_kw["resize"]),
                ]
            )
            self._transform = v2.Compose(
                [
                    v2.ToDtype(dtype=torch.float, scale=True),
                    v2.RandomHorizontalFlip(tf_kw["p_h_flip"]),  # convert to [-1,1]
                    v2.Normalize([0.5], [0.5]),
                ]
            )
        self._dataset.set_transform(lambda ex: self.transform(ex["image"]))

    def transform(
        self, images: list[Image.Image] | np.ndarray | torch.Tensor
    ) -> SampleType:
        """
        Transform batch of images.
        Args:
            images: batch of images

        Returns:
            SampleType

        """
        if self.tf_func:
            samples = self.tf_func(images)
        else:
            resized_images = torch.stack(self._to_resized_images(images), 0)
            samples = SampleType(
                images=self._transform(resized_images), raw_images=images
            )
        check_sample(samples)
        return samples

    def __getitem__(self, index) -> _T_co:
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)

    def __repr__(self):
        return self._dataset.__repr__()
    
    def show_images(self, indexes: Sequence[int]|range|Iterable, img_per_row=None, raw: bool = True):
        img_per_row = img_per_row or max(5, int(len(indexes) / 2))
        n_rows = ceil(len(indexes) / img_per_row)
        fig, axes = plt.subplots(
            n_rows, img_per_row, figsize=(img_per_row * 2, n_rows * 2)
        )
        axes = axes.flatten()

        images: list[Image.Image] | np.ndarray | torch.Tensor| None = None
        # noinspection PyTypeChecker
        samples: SampleType = self[indexes]
        if raw:
            images = samples["raw_images"]
        else:
            images = (samples["images"] / 2 + 0.5).permute(0, 2, 3, 1).numpy()

        for i, img in enumerate(images):
            if isinstance(img, Image.Image):
                assert raw==True
                title =  f"Size (WxH): {img.size}"
            else:
                assert raw==False and isinstance(img,np.ndarray)
                title = f"Size (HxW): {img.shape[:-1]}"
                
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")  # Hide axes ticks and labels
            
        for j in range(len(images), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    dataset = HugganSmithsonianButterfliesSubsetDataset()
    print(dataset)
    dataset.show_images(range(5))
    dataset.show_images(range(5), raw=False)

