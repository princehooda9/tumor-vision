import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss

torch.set_float32_matmul_precision('high')
# Import the dataloader we just built!
from data_loader import get_brats_dataloaders

# --- Update these specific parts in train_brats.py ---

class BraTSModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # New MONAI SwinUNETR Signature
        # It no longer needs img_size/roi_size in the init!
        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=48,    # Keep 48 for high complexity
            use_checkpoint=True,
            norm_name="instance",
            spatial_dims=3
        )
        
        self.loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # --- THE BULLETPROOF SHAPE FIX ---
        # If labels are 7D [1, 3, 3, 0, 96, 96, 155], we need to extract 
        # just the [Batch, Classes, Z, Y, X] parts we need.
        if len(labels.shape) > 5:
            # We take the first 3 classes and the last three 96x96x96 spatial dims
            labels = labels[:, :, -96:, -96:, -96:]
            # If there's still an extra dim in the middle, squeeze it
            while len(labels.shape) > 5:
                labels = labels.squeeze(2)

        # Do the same for images just in case
        if len(images.shape) > 5:
            images = images[:, :, -96:, -96:, -96:]
            while len(images.shape) > 5:
                images = images.squeeze(2)

        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # AdamW is currently the most stable optimizer for Transformers
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 2. DATA: Increase num_workers to 4 to speed up image processing
    # (Since you have an i5 with 10+ cores, this will stop the 'bottleneck' warning)
    train_loader = get_brats_dataloaders(data_dir="./data", cache_dir="./cache", batch_size=1)
    
    model = BraTSModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="brats-best",
        save_last=True,
        monitor="train_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=100,           # Full 100 epoch run
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        # limit_train_batches=None # Runs the whole dataset now
    )

    print("🏁 Omni-Med Going Live. Starting 100-epoch sprint...")
    # Tell the trainer to load the 'last' state from your checkpoints folder
    trainer.fit(
    model, 
    train_dataloaders=train_loader, 
    ckpt_path="./checkpoints/last.ckpt"
)