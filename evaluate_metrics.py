import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import SlidingWindowInferer
from data_loader import get_full_brain_dataloader
from train_brats import BraTSModel

def calculate_metrics(checkpoint_path="./checkpoints/brats-best.ckpt"):
    print("🧠 Loading Omni-Med for Quantitative Evaluation...")
    model = BraTSModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = get_full_brain_dataloader()
    
    # Initialize MONAI Metrics
    # include_background=False ensures we only score the 3 tumor classes, not the empty space
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5)

    print("⚡ Running evaluation (this will take a moment)...")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= 5: # Limit to 5 patients for a quick test
                break
                
            images = batch["image"].to(model.device)
            labels = batch["label"].to(model.device) 

            # 👻 THE GHOST DIMENSION FIX
            while len(labels.shape) > 5:
                labels = labels.squeeze(2)

            # 1. Predict
            outputs = inferer(images, model)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            # 2. Update Metrics
            dice_metric(y_pred=preds, y=labels)
            hd95_metric(y_pred=preds, y=labels)
            
            print(f"✅ Scored Patient {i+1}")

    # 3. Aggregate and Print Results
    mean_dice = dice_metric.aggregate().item()
    mean_hd95 = hd95_metric.aggregate().item()

    print("\n" + "="*40)
    print("🏆 OMNI-MED QUANTITATIVE RESULTS")
    print("="*40)
    print(f"Mean Dice Score (DSC):     {mean_dice:.4f} (Closer to 1.0 is better)")
    print(f"Hausdorff Distance (HD95): {mean_hd95:.2f} mm (Lower is better)")
    print("="*40)

if __name__ == "__main__":
    calculate_metrics()