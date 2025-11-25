import torch
from torch import nn
from torch.optim import AdamW
import lightning as L
import torchmetrics
from losses import WeightedFocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import CAFAMetrics

class LitMLPModule(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        taxonomy_num: int,   # Tổng số lượng mã taxonomy
        taxonomy_dim: int,   # Kích thước vector nhúng taxonomy
        hidden_dim: int,
        num_labels: int,
        learning_rate: float,
        term_weights: torch.Tensor, # Trọng số IA cho loss
        cfg,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['term_weights']) # Không save tensor lớn vào hparams
        self.term_weights = term_weights
        
        # 1. Taxonomy Branch
        # Embedding: [Num_Tax] -> [Tax_Dim]
        self.tax_embedding = nn.Embedding(taxonomy_num, taxonomy_dim)
        # Head nhỏ cho Tax
        self.tax_head = nn.Sequential(
            nn.Linear(taxonomy_dim, taxonomy_dim),
            nn.ReLU()
        )
        
        combined_dim = input_dim + taxonomy_dim
        
        self.main_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            nn.Linear(hidden_dim // 2, num_labels)
        )

        # Loss Function
        if cfg.loss == 'weighted_focal':
            self.criterion = WeightedFocalLoss(gamma=cfg.gamma, term_weights=self.term_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Validation Metrics
        self.val_f1 = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, average='micro')
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_labels)
        self.val_auprc = torchmetrics.AveragePrecision(task="multilabel", num_labels=num_labels)
        
        # CAFA F-max Metrics for validation
        self.val_fmax = CAFAMetrics()
        self.val_fmax_weighted = CAFAMetrics(ia=term_weights)


    def forward(self, x_seq, x_tax):
        tax_emb = self.tax_embedding(x_tax) # (B, Tax_Dim)
        tax_feat = self.tax_head(tax_emb)
        
        combined = torch.cat([x_seq, tax_feat], dim=1)
        
        # Main MLP
        logits = self.main_mlp(combined)
        return logits

    def _common_step(self, batch, stage: str):
        inputs = batch["embedding"]
        tax_ids = batch["taxonomy_idx"]
        labels = batch["labels"]

        logits = self(inputs, tax_ids)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, "train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._common_step(batch, "val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update validation metrics
        self.val_f1(preds, labels.int())
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auroc(preds, labels.int())
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auprc(preds, labels.int())
        self.log("val_auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)

        # Update CAFA metrics
        self.val_fmax.update(preds, labels)
        self.val_fmax_weighted.update(preds, labels)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log F-max
        fmax, prec, rec = self.val_fmax.compute()
        self.log("val_fmax", fmax, prog_bar=True)
        self.log("val_fmax_precision", prec)
        self.log("val_fmax_recall", rec)
        self.val_fmax.reset()

        # Compute and log weighted F-max
        fmax_w, prec_w, rec_w = self.val_fmax_weighted.compute()
        self.log("val_fmax_weighted", fmax_w, prog_bar=True)
        self.log("val_fmax_precision_weighted", prec_w)
        self.log("val_fmax_recall_weighted", rec_w)
        self.val_fmax_weighted.reset()
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch["embedding"]
        tax_ids = batch["taxonomy_idx"]
        protein_ids = batch["protein_id"] 
        
        logits = self(inputs, tax_ids)
        preds = torch.sigmoid(logits)
        return {"predictions": preds, "ids": protein_ids}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',      # We want to maximize Val F-max weighted
            factor=0.5,      # Reduce LR by half
            patience=2,      # If 2 consecutive epochs F-max doesn't improve -> Reduce LR
            verbose=True,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_fmax_weighted", # Monitor the weighted F-max
                "interval": "epoch",
                "frequency": 1,
            },
        }
