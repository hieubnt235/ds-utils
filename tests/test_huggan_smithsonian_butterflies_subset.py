import torch

from ds_utils.huggan_smithsonian_butterflies_subset import HugganSmithsonianButterfliesSubsetDataset, SampleType
import pytest
from PIL.Image import Image
@pytest.fixture
def dataset():
    return HugganSmithsonianButterfliesSubsetDataset()

@pytest.mark.parametrize("indexes",[[1,2,3], [3,5,6,6], range(5)])
def test_sample(dataset, indexes):
    sample: SampleType = dataset[indexes]
    assert isinstance(sample,dict)
    
    images: torch.Tensor = sample.get("images")
    raw_images: list[Image] = sample.get("raw_images")
    assert isinstance(raw_images,list) and isinstance(raw_images[0], Image)
    assert len(raw_images) == images.size(0)
    assert isinstance(images, torch.Tensor)
    assert torch.is_floating_point(images)
    assert len(s:=images.shape) == 4
    assert s[1] == 3
    assert s[0] == len(indexes)
    assert images.min()>=-1.0
    assert images.max()<=1.0