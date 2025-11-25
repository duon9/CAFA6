import pandas as pd
import numpy as np
import torch
import lightning as L
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from protein_dataset import LitProteinDataModule
from protein_module import LitMLPModule
from config import cfg
import gc
import os

def load_and_align_train_data():
    print("1. Loading Training Data...")
    
    # A. Load Sequence (để lấy EntryID làm gốc)
    print(f"   Loading Sequence from {cfg.sequence_path}...")
    seq_df = pd.read_parquet(cfg.sequence_path)
    
    # B. Load Taxonomy Train
    if os.path.exists(cfg.train_taxonomy_path):
        print(f"   Loading Train Taxonomy from {cfg.train_taxonomy_path}...")
        tax_df = pd.read_csv(cfg.train_taxonomy_path)
        
        # Kiểm tra cột
        if 'EntryID' not in tax_df.columns or 'taxonomyID' not in tax_df.columns:
            raise ValueError(f"File taxonomy {cfg.train_taxonomy_path} phải có cột 'EntryID' và 'taxonomyID'")
        seq_df = seq_df.merge(tax_df[['EntryID', 'taxonomyID']], on='EntryID', how='left')
        seq_df['taxonomyID'] = seq_df['taxonomyID'].fillna(-1)
    else:
        raise FileNotFoundError(f"Không tìm thấy file taxonomy train tại: {cfg.train_taxonomy_path}")

    # C. Load Embeddings
    print(f"   Loading embeddings from {cfg.embedding_path}...")
    embeddings = np.load(cfg.embedding_path)
    
    if len(seq_df) != len(embeddings):
        raise ValueError(f"Mismatch: Sequence DF ({len(seq_df)}) vs Embeddings ({len(embeddings)})")

    # D. Load Labels & Merge
    print(f"   Loading Labels from {cfg.label_path}...")
    lbl_df = pd.read_parquet(cfg.label_path)
    
    print("2. Merging Data & Labels...")
    # Giữ index gốc để filter embedding
    seq_df['original_index'] = seq_df.index
    
    # Merge Inner với Label (chỉ train những sample có nhãn)
    full_df = seq_df.merge(lbl_df, on='EntryID', how='inner')
    
    # Lọc embedding tương ứng
    valid_indices = full_df['original_index'].values
    filtered_embeddings = embeddings[valid_indices]
    
    print(f"   Final Train Shape: DF {full_df.shape}, Emb {filtered_embeddings.shape}")
    
    # Xác định label columns (loại bỏ các cột metadata)
    exclude_cols = ['EntryID', 'taxonomyID', 'original_index', 'seq'] 
    label_cols = [c for c in full_df.columns if c in lbl_df.columns and c not in exclude_cols]
    
    # E. Build Taxonomy Map
    print("3. Building Taxonomy Map...")
    unique_taxonomies = full_df['taxonomyID'].unique()
    
    # Index 0 dành cho <unknown>
    taxonomy_map = {'<unknown>': 0}
    # Bắt đầu đánh số từ 1
    current_idx = 1
    for val in unique_taxonomies:
        if val != -1: # Bỏ qua giá trị fillna ban đầu nếu có
            taxonomy_map[val] = current_idx
            current_idx += 1
            
    num_taxonomies = len(taxonomy_map)
    print(f"   Found {num_taxonomies - 1} unique taxonomies (plus unknown).")
    
    return full_df, filtered_embeddings, label_cols, taxonomy_map, num_taxonomies

def load_test_data(taxonomy_map):
    print("   Loading Test Data...")
    if not (os.path.exists(cfg.test_path) and os.path.exists(cfg.test_embedding_path)):
        print("   Test files not found. Skipping prediction.")
        return None, None

    test_df = pd.read_csv(cfg.test_path)
    test_emb = np.load(cfg.test_embedding_path)
    
    # Load Test Taxonomy và Merge
    if os.path.exists(cfg.test_taxonomy_path):
        print(f"   Merging Test Taxonomy from {cfg.test_taxonomy_path}...")
        test_tax_df = pd.read_csv(cfg.test_taxonomy_path)
        
        if 'EntryID' in test_tax_df.columns and 'taxonomyID' in test_tax_df.columns:
            test_df = test_df.merge(test_tax_df[['EntryID', 'taxonomyID']], on='EntryID', how='left')
            test_df['taxonomyID'] = test_df['taxonomyID'].fillna(-1) 
        else:
            test_df['taxonomyID'] = -1
    else:
        test_df['taxonomyID'] = -1
        
    return test_df, test_emb

def load_ia_weights(ia_path, label_cols):
    print("4. Loading IA Weights...")
    if not os.path.exists(ia_path):
        print("   Warning: ia.csv not found, using equal weights.")
        return None
        
    ia_df = pd.read_csv(ia_path)
    if 'term' not in ia_df.columns: ia_df.rename(columns={ia_df.columns[0]: 'term'}, inplace=True)
    if 'score' not in ia_df.columns: ia_df.rename(columns={ia_df.columns[1]: 'score'}, inplace=True)
    
    weight_dict = dict(zip(ia_df['term'], ia_df['score']))
    
    weights = []
    for col in label_cols:
        weights.append(weight_dict.get(col, 1.0))
        
    return torch.tensor(weights, dtype=torch.float32)

def main():
    L.seed_everything(42)
    
    # 1. Load Train Data (Đã sửa logic merge taxonomy)
    full_df, embeddings, label_cols, tax_map, num_tax = load_and_align_train_data()
    
    # 2. Load Weights
    term_weights = load_ia_weights(cfg.ia_path, label_cols)
    
    # 3. Split Train/Val
    train_idx, val_idx = train_test_split(np.arange(len(full_df)), test_size=0.2, random_state=42)
    
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)
    train_emb = embeddings[train_idx]
    val_emb = embeddings[val_idx]
    
    del full_df, embeddings
    gc.collect()

    # 4. Load Test Data
    test_df, test_emb = load_test_data(tax_map)

    # 5. Setup Data Module
    data_module = LitProteinDataModule(
        train_df=train_df,
        train_emb=train_emb,
        val_df=val_df,
        val_emb=val_emb,
        test_df=test_df,
        test_emb=test_emb,
        label_columns=label_cols,
        taxonomy_map=tax_map,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    data_module.setup() 
    
    # 6. Setup Model
    print(f"5. Init Model (SeqDim: {cfg.input_dim}, TaxDim: {cfg.taxonomy_dim}, Classes: {len(label_cols)})...")
    model = LitMLPModule(
        input_dim=cfg.input_dim,
        taxonomy_num=num_tax,
        taxonomy_dim=cfg.taxonomy_dim,
        hidden_dim=cfg.hidden_dim,
        num_labels=len(label_cols),
        learning_rate=cfg.learning_rate,
        term_weights=term_weights,
        cfg=cfg
    )
    
    # 7. Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-multimodal",
        save_top_k=1,
        monitor="val_f1",
        mode="max"
    )

    early_stopping = EarlyStopping(monitor="val_f1", patience=cfg.patience, mode="max")
    mlf_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=cfg.tracking_uri)

    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto", 
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=mlf_logger
    )
    
    print("Start Training...")
    trainer.fit(model, data_module)
    
    # 8. Predict
    if test_df is not None:
        print("Predicting on Test Set...")
        predictions = trainer.predict(model, dataloaders=data_module.test_dataloader(), ckpt_path='best')
        
        all_ids = []
        all_scores = []
        for batch in predictions:
            all_ids.extend(batch['ids'])
            all_scores.append(batch['predictions'])
            
        preds_tensor = torch.cat(all_scores, dim=0).cpu().numpy()
        
        preds_df = pd.DataFrame(preds_tensor, columns=label_cols)
        preds_df['EntryID'] = all_ids
        
        melted = preds_df.melt(id_vars='EntryID', var_name='term', value_name='score')
        melted = melted[melted['score'] > 0.1] 
        
        melted.to_csv(cfg.output, sep='\t', index=False, header=False)
        print(f"Done. Saved to {cfg.output}")

if __name__ == "__main__":
    main()