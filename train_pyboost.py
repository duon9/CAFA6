import pandas as pd
import numpy as np
import gc
import os
import joblib
from sklearn.model_selection import train_test_split
from py_boost import GradientBoosting
from py_boost.multioutput.sketching import RandomProjectionSketch

# Import config mới
from config_pyboost import cfg

# --- Tái sử dụng các hàm nạp dữ liệu từ train.py gốc ---
# (Các hàm này không phụ thuộc vào PyTorch)

def load_and_align_train_data():
    print("1. Loading Training Data...")
    print(f"   Loading Sequence from {cfg.sequence_path}...")
    seq_df = pd.read_parquet(cfg.sequence_path)

    # Taxonomy data is no longer used.

    print(f"   Loading embeddings from {cfg.embedding_path}...")
    embeddings = np.load(cfg.embedding_path)
    if len(seq_df) != len(embeddings):
        raise ValueError(f"Mismatch: Sequence DF ({len(seq_df)}) vs Embeddings ({len(embeddings)})")

    print(f"   Loading Labels from {cfg.label_path}...")
    lbl_df = pd.read_parquet(cfg.label_path)
    # Removed 'taxonomyID' from exclude_cols
    exclude_cols = ['EntryID', 'original_index', 'seq']
    label_cols = [c for c in lbl_df.columns if c not in exclude_cols and c != 'EntryID']

    print("2. Merging Data & Labels...")
    seq_df['original_index'] = np.arange(len(seq_df))
    full_df = seq_df.merge(lbl_df, on='EntryID', how='inner')
    
    valid_indices = full_df['original_index'].values
    filtered_embeddings = embeddings[valid_indices]
    
    print(f"   Final Train Shape: DF {full_df.shape}, Emb {filtered_embeddings.shape}")
    return full_df, filtered_embeddings, label_cols

def load_test_data():
    print("   Loading Test Data...")
    if not (os.path.exists(cfg.test_path) and os.path.exists(cfg.test_embedding_path)):
        print("   Test files not found. Skipping prediction.")
        return None, None
    test_df = pd.read_csv(cfg.test_path).rename(columns={'id': 'EntryID'})
    test_emb = np.load(cfg.test_embedding_path)

    # Taxonomy data is no longer used.
    
    return test_df, test_emb

# --- Hàm Main mới cho Py-Boost ---

def main():
    np.random.seed(42)
    
    # 1. Load Data
    full_df, embeddings, label_cols = load_and_align_train_data()

    # 2. Split Train/Val
    train_idx, val_idx = train_test_split(np.arange(len(full_df)), test_size=0.15, random_state=42)

    # 3. Chuẩn bị dữ liệu cho Py-Boost
    print("3. Preparing data for Py-Boost...")

    # Feature matrix X is now just the embeddings
    X_train = embeddings[train_idx].astype('float32')
    X_val = embeddings[val_idx].astype('float32')

    # Get labels
    y_train = full_df.iloc[train_idx][label_cols].values.astype('int32')
    y_val = full_df.iloc[val_idx][label_cols].values.astype('int32')

    print(f"   Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val shapes:   X={X_val.shape}, y={y_val.shape}")

    del full_df, embeddings
    gc.collect()

    # 4. Cấu hình và Huấn luyện mô hình Py-Boost
    print("\n4. Configuring and Training Py-Boost model...")
    if cfg.sketch_type == 'random_projection':
        sketch = RandomProjectionSketch(cfg.sketch_dim)
    else:
        # Có thể thêm các loại sketch khác ở đây
        raise ValueError(f"Unsupported sketch_type: {cfg.sketch_type}")

    model = GradientBoosting(
        loss=cfg.pyboost_loss,
        ntrees=cfg.ntrees,
        lr=cfg.lr,
        es=cfg.es,
        max_depth=cfg.max_depth,
        lambda_l2=cfg.lambda_l2,
        use_hess=cfg.use_hess,
        multioutput_sketch=sketch,
        verbose=10, # In log mỗi 10 cây
    )

    model.fit(
        X_train, y_train,
        eval_sets=[{'X': X_val, 'y': y_val}]
    )

    # 5. Lưu lại model
    print("\n5. Saving model...")
    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(model, cfg.model_save_path)
    print(f"   Model saved to {cfg.model_save_path}")

    del X_train, y_train, X_val, y_val
    gc.collect()

    # 6. Dự đoán trên tập test
    print("\n6. Predicting on Test Set...")
    test_df, test_emb = load_test_data()
    if test_df is not None:
        # Feature matrix X_test is now just the test embeddings
        X_test = test_emb.astype('float32')

        print(f"   Test shape: X={X_test.shape}")
        
        # Dự đoán xác suất
        predictions = model.predict(X_test)
        
        # Chuyển thành DataFrame và xử lý submission
        preds_df = pd.DataFrame(predictions, columns=label_cols)
        preds_df['EntryID'] = test_df['EntryID']
        
        # Chuyển từ dạng wide sang long và lọc theo ngưỡng
        melted = preds_df.melt(id_vars='EntryID', var_name='term', value_name='score')
        melted = melted[melted['score'] > cfg.prediction_threshold]
        
        melted.to_csv(cfg.output, sep='\t', index=False, header=False)
        print(f"\nDone. Submission file saved to {cfg.output}")

if __name__ == "__main__":
    main()
