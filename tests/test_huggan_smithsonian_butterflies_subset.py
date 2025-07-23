from math import isclose

import torch

from ds_utils.huggan_smithsonian_butterflies_subset import Dataset, SampleType
import pytest
from PIL.Image import Image
from torchvision.transforms import v2
@pytest.fixture
def dataset():
    return Dataset()

@pytest.fixture
def split_datasets()->tuple[Dataset,Dataset]:
    return Dataset.split(0.8)

@pytest.mark.parametrize("indexes",[[1,2,3], [3,5,6,6], range(2),*[i for i in range(2)]])
def test_sample(dataset, indexes):
    sample: SampleType = dataset[indexes]
    assert isinstance(sample,dict)
    
    images: torch.Tensor = sample.get("images")
    raw_images: list[Image]|Image = sample.get("raw_images")
    assert isinstance(images, torch.Tensor)
    assert torch.is_floating_point(images)
    assert images.min()>=-1.0 or isclose(images.min(), -1.0, abs_tol=1e-6)
    assert images.max()<=1.0 or isclose(images.max(), 1.0, abs_tol=1e-6)
    
    
    if isinstance(raw_images,list):
        assert len(images.shape)==4
        assert len(raw_images) == images.size(0) == len(indexes)
        raw_image = raw_images[0]
        image = images[0]
    else:
        assert len(images.shape) == 3
        raw_image = raw_images
        image = images
    
    assert  isinstance(raw_image, Image)
    assert image.shape[0] == 3
    assert image.shape[1:] == dataset.tf_kw["resize"]

@pytest.mark.parametrize("idx",[i for i in range (2)])
@pytest.mark.parametrize("resize",[(50,50), (88,88)])
@pytest.mark.parametrize("train_ratio",[0.7,0.53])
def test_split(idx, resize, train_ratio):
    train_ds, val_ds = Dataset.split(train_ratio,True,v2.Resize(resize))
    assert (len(train_ds)/(len(train_ds)+len(val_ds)) - train_ratio) < 0.001
    
    assert train_ds.mode=="train" and val_ds.mode=="val"
    
    assert isinstance(train_img:=train_ds[idx]["images"],torch.Tensor) and len(ts:=train_img.shape)==3 and ts[0]==3
    assert ts[1:] == resize
    
    assert isinstance(imgs:=val_ds[idx]["images"],torch.Tensor) and len(vs:=imgs.shape)==3 and vs[0]==3
    assert vs[1:] == val_ds.tf_kw["resize"]
    
    