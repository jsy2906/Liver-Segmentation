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


def preprocess(# niftifile_dir,
                image_path, label_path, test_size=0.3,
                pixdim=[1.5, 1.5, 1.0], a_min=-200, a_max=200, 
                train_spatial_size=[128,128,64], valid_spatial_size=[128,128,64],
                train_batch=2, valid_batch=2):

    # image_path = niftifile_dir + '/images/not_empty/*'
    # label_path = niftifile_dir + '/labels/not_empty/*'

    xtrain, xvalid, ytrain, yvalid = train_test_split(glob(image_path), 
                                                    glob(label_path), 
                                                    test_size=test_size, random_state=0
                                                    )

    train = [{'image' : image_name, 'label': label_name} for image_name, label_name in zip(sorted(xtrain), sorted(ytrain))]
    valid = [{'image' : image_name, 'label': label_name} for image_name, label_name in zip(sorted(xvalid), sorted(yvalid))]

    # Transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            RandFlipd(keys=['image', 'label'], spatial_axis=[0], prob=0.3),
            RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3),
            Resized(keys=["image", "label"], spatial_size=train_spatial_size),   
            ToTensord(keys=["image", "label"]),
        ]
    )

    valid_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=["image", "label"], spatial_size=valid_spatial_size),   
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_ds = monai.data.CacheDataset(data=train, transform=train_transforms)
    train_dl = monai.data.DataLoader(train_ds, batch_size=train_batch)

    valid_ds = monai.data.CacheDataset(data=valid, transform=valid_transforms)
    valid_dl = monai.data.DataLoader(valid_ds, batch_size=valid_batch)

    return train_dl, valid_dl
