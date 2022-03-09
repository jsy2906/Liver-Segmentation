import random
import torch

import monai
from glob import glob

from monai.transforms import (
                                LoadImaged,
                                AddChanneld,
                                Spacingd,   # affine field를 새로 씀
                                Orientationd,   # 방향전환
                                ScaleIntensityRanged,   # output intensity 범위 조정
                                CropForegroundd,
                                RandFlipd,
                                RandRotate90d,
                                Resized,
                                ToTensord,
                                Compose,
                                )

from sklearn.model_selection import train_test_split
import nibabel as nib
import numpy as np

random_seed = 0
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)

def preprocess(image_path, label_path, include_not_empty=True, test_size=0.3,
                pixdim=[1.5, 1.5, 1.0], a_min=-200, a_max=200, 
                train_spatial_size=[128,128,64], valid_spatial_size=[128,128,64],
                include_back=False,
                train_batch=2, valid_batch=2, shuffle=True):

    image_path = image_path + '/*'
    label_path = label_path + '/*'

    xtrain, xvalid, ytrain, yvalid = train_test_split(glob(image_path), 
                                                    glob(label_path), 
                                                    test_size=test_size, random_state=0,
                                                    )

    train = [{'image' : image_name, 'label': label_name} for image_name, label_name in zip(sorted(xtrain), sorted(ytrain))]
    if include_not_empty: not_empty = find_empty(train)
    valid = [{'image' : image_name, 'label': label_name} for image_name, label_name in zip(sorted(xvalid), sorted(yvalid))]

    train_ds = monai.data.CacheDataset(data=train, transform=get_transform(train, pixdim=pixdim, 
                                                                    a_min=a_min, a_max=a_max, train_spatial_size=train_spatial_size, 
                                                                    include_back=include_back))
    train_dl = monai.data.DataLoader(train_ds, batch_size=train_batch, shuffle=shuffle)
    if include_not_empty: 
        not_empty_ds = monai.data.CacheDataset(data=not_empty, transform=get_transform(not_empty, pixdim=pixdim, 
                                                                    a_min=a_min, a_max=a_max, train_spatial_size=train_spatial_size, 
                                                                    include_back=include_back))
        not_empty_dl = monai.data.DataLoader(not_empty_ds, batch_size=train_batch, shuffle=shuffle)
    if include_not_empty:
        train_list = []
        train_list.extend(train_dl)
        train_list.extend(not_empty_dl)
        train_dl = train_list

    valid_ds = monai.data.CacheDataset(data=valid, transform=get_transform(valid, pixdim=pixdim, 
                                                                    a_min=a_min, a_max=a_max, train_spatial_size=valid_spatial_size, 
                                                                    include_back=include_back))
    valid_dl = monai.data.DataLoader(valid_ds, batch_size=valid_batch, shuffle=shuffle)

    return train_dl, valid_dl


def get_transform(data, pixdim=[1.5, 1.5, 1.0], a_min=-200, a_max=200, 
                train_spatial_size=[128,128,64], valid_spatial_size=[128,128,64],
                include_back=True):

    data=str(data)
    train_transform = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            RandFlipd(keys=['image', 'label'], spatial_axis=[0], prob=0.3),
            RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3),
            Resized(keys=["image", "label"], spatial_size=train_spatial_size),   
            ToTensord(keys=["image", "label"]),
        ]

    transform = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=["image", "label"], spatial_size=train_spatial_size),   
            ToTensord(keys=["image", "label"]),
        ]

    valid_transform = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            Resized(keys=["image", "label"], spatial_size=valid_spatial_size),   
            ToTensord(keys=["image", "label"]),
        ]

    if not include_back:
        train_transform.append(CropForegroundd(keys=['image', 'label'], source_key='image'))
        valid_transform.append(CropForegroundd(keys=['image', 'label'], source_key='image'))

    if data=='train':
        return Compose(train_transform)

    elif data=='valid':
        return Compose(valid_transform)

    else:
        return Compose(transform)


def find_empty(datas):
    not_empty = []
    for data in datas:
        label = data['label']
        nifti_file = nib.load(label)
        fdata = nifti_file.get_fdata()
        np_unique = np.unique(fdata)
        if len(np_unique) > 1:
            not_empty.append(data)

    return not_empty
