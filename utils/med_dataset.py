import math
import os
import torch
import numpy as np
from monai import data, transforms as med
from monai.data import load_decathlon_datalist
import PIL.Image as PImage
from torch.utils.data import Dataset


class MedicalDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.sample_list = os.listdir(self._base_dir)
        self.transform = transform
        print("total {}".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        img = PImage.open(os.path.join(self._base_dir, case)).convert('RGB')
        aug = self.transform(img)
        return aug


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(data_dir, size):
    datalist_json = os.path.join(data_dir, "dataset.json")
    train_transform = med.Compose(
    [
        med.LoadImaged(keys=["image"], allow_missing_keys=True),
        med.AddChanneld(keys=["image"], allow_missing_keys=True),
        med.Orientationd(keys=["image"], axcodes="RAS", allow_missing_keys=True),
        med.Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear", allow_missing_keys=True),
        med.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        med.CropForegroundd(keys=["image"], source_key="image", allow_missing_keys=True),
        med.SpatialPadd(keys=["image"], spatial_size=(size, size, size), mode='constant'),
        med.RandCropByPosNegLabeld(
            spatial_size=(size, size, size),
            keys=["image"],
            label_key="image",
            pos=1,
            neg=0,
            num_samples=4,
        ),
        med.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
        med.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
        med.RandFlipd(keys=["image"], prob=0.1, spatial_axis=2),
        med.ToTensord(keys=["image"]),
    ])
    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    train_ds = data.CacheNTransDataset(data=datalist, transform=train_transform, cache_n_trans=6, cache_dir="./cache_dataset")
    return train_ds



