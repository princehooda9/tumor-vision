import os
import glob
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    MapTransform,
    Orientationd,
    Spacingd,
    ConcatItemsd,
)
from monai.data import PersistentDataset, DataLoader, Dataset

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # [1, 2, 4] -> [TC, WT, ET]
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 4), d[key] == 2))
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d

def get_brats_dataloaders(data_dir="./data", cache_dir="./cache", batch_size=1):
    patient_folders = [f for f in sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*"))) if os.path.isdir(f)]
    
    data_dicts = []
    for folder in patient_folders:
        pid = os.path.basename(folder)
        # Identify individual files
        t1 = glob.glob(os.path.join(folder, "*_t1.nii.gz"))
        t1ce = glob.glob(os.path.join(folder, "*_t1ce.nii.gz"))
        t2 = glob.glob(os.path.join(folder, "*_t2.nii.gz"))
        flair = glob.glob(os.path.join(folder, "*_flair.nii.gz"))
        seg = glob.glob(os.path.join(folder, "*_seg.nii.gz"))

        if all([t1, t1ce, t2, flair, seg]):
            data_dicts.append({
                "t1": t1[0], "t1ce": t1ce[0], "t2": t2[0], "flair": flair[0],
                "label": seg[0],
            })

    # We list all 4 MRI keys + the label key for spatial transforms
    img_keys = ["t1", "t1ce", "t2", "flair"]
    all_keys = img_keys + ["label"]

    train_transform = Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
        
        # Move the crop BEFORE we concatenate to keep metadata intact
        RandSpatialCropd(keys=all_keys, roi_size=[96, 96, 96], random_size=False),
        
        # Now concatenate and normalize
        ConcatItemsd(keys=img_keys, name="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


    # Start with a small subset to verify the cache works
    from monai.data import CacheDataset, DataLoader, Dataset

# Change PersistentDataset to CacheDataset
# We will cache 100 patients in RAM. The rest will be loaded from disk on the fly.
    train_ds = Dataset(data=data_dicts, transform=train_transform)

# 2. Use your i5 cores to process data on the fly
    train_loader = DataLoader(
    train_ds, 
    batch_size=1, 
    shuffle=True, 
    num_workers=4,     # Using 6 Performance cores for heavy lifting
    pin_memory=True    # Faster transfer to your 4060
)

    return train_loader

def get_full_brain_dataloader(data_dir="./data"):
    """Loads the entire uncut brain for Sliding Window Inference."""
    patient_folders = [f for f in sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*"))) if os.path.isdir(f)]
    
    data_dicts = []
    for folder in patient_folders:
        t1 = glob.glob(os.path.join(folder, "*_t1.nii.gz"))
        t1ce = glob.glob(os.path.join(folder, "*_t1ce.nii.gz"))
        t2 = glob.glob(os.path.join(folder, "*_t2.nii.gz"))
        flair = glob.glob(os.path.join(folder, "*_flair.nii.gz"))
        seg = glob.glob(os.path.join(folder, "*_seg.nii.gz"))

        if all([t1, t1ce, t2, flair, seg]):
            data_dicts.append({
                "t1": t1[0], "t1ce": t1ce[0], "t2": t2[0], "flair": flair[0],
                "label": seg[0],
            })

    img_keys = ["t1", "t1ce", "t2", "flair"]
    all_keys = img_keys + ["label"]

    # 🚨 Notice: NO RandSpatialCropd! We keep the massive 240x240x155 size.
    test_transform = Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        # Spacing is crucial so the voxel dimensions match training
        Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
        ConcatItemsd(keys=img_keys, name="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    test_ds = Dataset(data=data_dicts, transform=test_transform)
    # batch_size MUST be 1 for full brain volumes
    return DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)

if __name__ == "__main__":
    loader = get_brats_dataloaders()
    print("Testing data loader...")
    for batch_data in loader:
        print(f"Image shape: {batch_data['image'].shape}") # Should be [1, 4, 96, 96, 96]
        print(f"Label shape: {batch_data['label'].shape}") # Should be [1, 3, 96, 96, 96]
        break