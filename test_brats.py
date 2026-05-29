import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
from monai.networks.nets import SwinUNETR
import pytorch_lightning as pl
from monai.inferers import SlidingWindowInferer

# Import your new full-brain loader
from data_loader import get_full_brain_dataloader
from train_brats import BraTSModel

def run_evaluation(checkpoint_path="./checkpoints/brats-best.ckpt"):
    print("🧠 Loading trained Omni-Med model...")
    model = BraTSModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print("📂 Fetching raw, uncropped full brain scan...")
    # Call the new function we just added to data_loader.py
    data_loader = get_full_brain_dataloader()
    
    # Grab the first patient
    batch = next(iter(data_loader))
    images = batch["image"].to(model.device)
    labels = batch["label"]

    print(f"📏 Input Shape into Model: {images.shape}") # Should be ~ [1, 4, 240, 240, 155]

    print("⚡ Running Sliding Window Inference across the ENTIRE brain...")
    # Overlap of 0.5 stitches the 96x96x96 chunks together seamlessly
    inferer = SlidingWindowInferer(
        roi_size=(96, 96, 96), 
        sw_batch_size=4,       
        overlap=0.5            
    )

    with torch.no_grad():
        outputs = inferer(images, model)
        outputs = torch.sigmoid(outputs)
        preds = (outputs > 0.2).float() 
        
    print(f"✅ Full Brain Processed! Final Prediction Shape: {preds.shape}")

    # ==========================================
    # 💾 EXPORT NIFTI
    # ==========================================
    pred_data = preds[0].cpu().numpy()
    prediction_count = np.sum(pred_data)
    print(f"📊 Total Voxel Detections in Prediction: {int(prediction_count)}")

    pred_file = "omni_med_prediction.nii.gz"
    gt_file = "omni_med_ground_truth.nii.gz"
    flair_file = "omni_med_flair.nii.gz"

    for file_path in [pred_file, gt_file, flair_file]:
        if os.path.exists(file_path):
            os.remove(file_path)

    if prediction_count == 0:
        print("❌ Warning: 0 tumor voxels predicted.")
    else:
        # Save Prediction
        final_3d_volume = np.zeros(pred_data.shape[1:], dtype=np.uint8)
        final_3d_volume[pred_data[0] == 1] = 1 # WT
        final_3d_volume[pred_data[1] == 1] = 2 # TC
        final_3d_volume[pred_data[2] == 1] = 3 # ET
        nib.save(nib.Nifti1Image(final_3d_volume, affine=np.eye(4)), pred_file)
        
        # Save Ground Truth
        gt_data = labels[0].cpu().numpy()
        gt_volume = np.zeros(gt_data.shape[1:], dtype=np.uint8)
        gt_volume[gt_data[0] == 1] = 1
        gt_volume[gt_data[1] == 1] = 2
        gt_volume[gt_data[2] == 1] = 3
        nib.save(nib.Nifti1Image(gt_volume, affine=np.eye(4)), gt_file)

        # Save Brain Context
        flair_data = images[0, 3].cpu().numpy()
        flair_normalized = ((flair_data - flair_data.min()) / (flair_data.max() - flair_data.min()) * 255).astype(np.uint8)
        nib.save(nib.Nifti1Image(flair_normalized, affine=np.eye(4)), flair_file)

        print("📦 DONE: All NIfTI files exported successfully.")

if __name__ == "__main__":
    run_evaluation()