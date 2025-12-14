import pandas as pd
import numpy as np
import gc
import os
import joblib
from sklearn.model_selection import KFold
from py_boost import SketchBoost
import pyarrow as pa
import pyarrow.parquet as pq
import mlflow
from datetime import datetime
from py_boost import GradientBoosting
from py_boost.gpu.losses import WeightedBCELoss
from config import cfg
import cupy as cp
    

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
    exclude_cols = ['EntryID', 'original_index', 'seq']
    all_label_cols = [c for c in lbl_df.columns if c not in exclude_cols and c != 'EntryID']

    if cfg.top_k is not None and cfg.top_k > 0:
        print(f"   Filtering for top {cfg.top_k} most frequent labels...")
        label_counts = lbl_df[all_label_cols].sum().sort_values(ascending=False)
        top_k_labels = label_counts.head(cfg.top_k).index.tolist()
        print(f"   Original number of labels: {len(all_label_cols)}")
        print(f"   Number of labels after top-k filtering: {len(top_k_labels)}")
        label_cols = top_k_labels
    else:
        print("   Skipping top-k label filtering. Using all labels.")
        label_cols = all_label_cols

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

    
    return test_df, test_emb

def calculate_weight(y_true, method='inverse_freq', clip_min=0.1, clip_max=10.0):
    y_true = np.array(y_true)
    num_labels = y_true.shape[1]

    weight = np.zeros((num_labels, 2), dtype=np.float32)

    for j in range(num_labels):
        pos_count = np.sum(y_true[:, j] == 1)
        neg_count = np.sum(y_true[:, j] == 0)

        if method == 'inverse_freq':
            weight[j,1] = 1.0 / max(pos_count, 1)   
            weight[j,0] = 1.0 / max(neg_count, 1)   
        elif method == 'balanced':
            total = pos_count + neg_count
            weight[j,1] = total / (2 * max(pos_count, 1))
            weight[j,0] = total / (2 * max(neg_count, 1))
        else:
            raise ValueError("method must be 'inverse_freq' or 'balanced'")

    weight = np.clip(weight, clip_min, clip_max)
    return weight


def main():
    np.random.seed(42)
    
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    run_name = f"pyboost_5fold_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    full_df, embeddings, label_cols = load_and_align_train_data()
    X = embeddings.astype('float32')
    y = full_df[label_cols].values.astype('int32')

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = []
    val_f1_scores = []

    with mlflow.start_run(run_name=run_name) as parent_run:
        print(f"MLflow Run ID: {parent_run.info.run_id}")
        mlflow.log_params({k: v for k, v in cfg.__dict__.items() if not k.startswith('__')})

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            with mlflow.start_run(nested=True) as child_run:
                print(f"\n--- Fold {fold+1}/{n_splits} ---")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                weight = calculate_weight(y_train)
                weight[:10, :] = 1.0
                weight = cp.array(weight)

                print(f"   Train shapes: X={X_train.shape}, y={y_train.shape}")
                print(f"   Val shapes:   X={X_val.shape}, y={y_val.shape}")

                print("\n4. Configuring and Training Py-Boost model...")
                model = SketchBoost(loss = 'bce',
                                    metric = 'f1',
                                    ntrees=30000,
                                    lr=.01,
                                    verbose=100,
                                    es=200,
                                    lambda_l2=10,
                                    subsample=.8,
                                    colsample=.8,
                                    min_data_in_leaf=10,
                                    min_gain_to_split=0, 
                                    max_bin=256,
                                    sketch_outputs=3,
                                    sketch_method='proj',
                                    use_hess=False,
                                    max_depth=6)

                model.fit(
                    X_train, y_train,
                    eval_sets=[{'X': X_val, 'y': y_val}]
                )
                                
                cbs = model.callbacks.callbacks
                es = next(cb for cb in cbs if cb.__class__.__name__ == "EarlyStopping")
                best_f1 = es.best_score
                print(f"   Fold {fold+1} Best F1 Score: {best_f1:.4f}")
                val_f1_scores.append(best_f1)
                mlflow.log_metric("val_f1", best_f1)
                
                joblib.dump(model, f"model_fold_{fold+1}.pkl")
                # mlflow.log_artifact(f"model_fold_{fold+1}.pkl", artifact_path=f"model_fold_{fold+1}")
                # os.remove(f"model_fold_{fold+1}.pkl")
                print(f"   Model for fold {fold+1} logged to MLflow artifacts.")
                models.append(model)

                del X_train, y_train, X_val, y_val, model
                gc.collect()
                
            break
        # Log summary metrics vào run cha
        avg_f1 = np.mean(val_f1_scores)
        std_f1 = np.std(val_f1_scores)
        print(f"\n--- Cross-Validation Summary ---")
        print(f"   Average F1: {avg_f1:.4f}")
        print(f"   Std Dev F1: {std_f1:.4f}")
        mlflow.log_metric("avg_val_f1", avg_f1)
        mlflow.log_metric("std_val_f1", std_f1)

    del X, y, full_df, embeddings
    gc.collect()

    # model1 = joblib.load('model_fold_1.pkl')
    # model2 = joblib.load('model_fold_2.pkl')
    # model3 = joblib.load('model_fold_3.pkl')
    # model4 = joblib.load('model_fold_4.pkl')
    # model5 = joblib.load('model_fold_5.pkl')

    # models.extend([model1, model2, model3, model4, model5])
    # models.append(model1)
    # 5. Dự đoán trên tập test
    print("\n6. Predicting on Test Set...")
    test_df, test_emb = load_test_data()
    if test_df is not None:
        # Feature matrix X_test is now just the test embeddings
        X_test = test_emb.astype('float32')

        print(f"   Test shape: X={X_test.shape}")

        # Xử lý và ghi dự đoán theo từng chunk để tiết kiệm RAM
        print(f"Writing predictions incrementally to {cfg.output}...")
        chunk_size = 5000
        writer = None
        try:
            for i in range(0, len(X_test), chunk_size):
                print(f"  Processing chunk {i // chunk_size + 1}/{(len(X_test) - 1) // chunk_size + 1}...")
                chunk_end = min(i + chunk_size, len(X_test))
                
                X_chunk = X_test[i:chunk_end]
                ids_chunk = test_df['EntryID'][i:chunk_end].values
                
                # Dự đoán trên chunk hiện tại và lấy trung bình từ các model
                all_preds = []
                for fold_model in models:
                    all_preds.append(fold_model.predict(X_chunk))
                
                # Trung bình kết quả từ 5 models
                preds_chunk = np.mean(all_preds, axis=0)
                
                # Tạo DataFrame cho chunk này
                preds_df_chunk = pd.DataFrame(preds_chunk, columns=label_cols)
                preds_df_chunk['EntryID'] = ids_chunk
                
                # Chuyển sang dạng long, lọc và chuẩn bị ghi
                melted_chunk = preds_df_chunk.melt(id_vars='EntryID', var_name='term', value_name='score')
                melted_chunk = melted_chunk[melted_chunk['score'] > cfg.prediction_threshold]
                
                table_chunk = pa.Table.from_pandas(melted_chunk, preserve_index=False)
                
                if writer is None:
                    writer = pq.ParquetWriter(cfg.output, table_chunk.schema, compression='snappy')
                writer.write_table(table_chunk)
        finally:
            if writer:
                writer.close()
        print(f"\nDone. Submission file saved to {cfg.output}")

if __name__ == "__main__":
    main()
