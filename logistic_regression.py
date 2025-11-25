
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os
from config import cfg
import gc

def calculate_fmax(targets, preds, ia=None):
    """
    Calculate F-max score, precision, and recall, mimicking the CAFAMetrics logic.
    """
    fmax = 0.0
    prec = 0.0
    rec = 0.0
    best_tau = 0.0

    for t in range(1, 100):
        tau = t / 100.0
        p = preds >= tau

        if ia is None:
            tp = np.sum(np.logical_and(p, targets))
            fp = np.sum(np.logical_and(p, np.logical_not(targets)))
            fn = np.sum(np.logical_and(np.logical_not(p), targets))
        else:
            tp = np.sum(np.logical_and(p, targets) * ia)
            fp = np.sum(np.logical_and(p, np.logical_not(targets)) * ia)
            fn = np.sum(np.logical_and(np.logical_not(p), targets) * ia)

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        if precision + recall > 0:
            f = 2 * (precision * recall) / (precision + recall)
            if f > fmax:
                fmax = f
                prec = precision
                rec = recall
                best_tau = tau
    
    return fmax, prec, rec, best_tau

def load_data():
    print("1. Loading Training Data...")
    
    # Load sequence to get EntryID order
    print(f"   Loading Sequence from {cfg.sequence_path}...")
    seq_df = pd.read_parquet(cfg.sequence_path)[['EntryID']]

    # Load Embeddings
    print(f"   Loading embeddings from {cfg.embedding_path}...")
    embeddings = np.load(cfg.embedding_path)
    
    if len(seq_df) != len(embeddings):
        raise ValueError(f"Mismatch: Sequence DF ({len(seq_df)}) vs Embeddings ({len(embeddings)})")

    # Load Labels
    print(f"   Loading Labels from {cfg.label_path}...")
    lbl_df = pd.read_parquet(cfg.label_path)
    
    print("2. Merging Data & Labels...")
    # Keep original index to filter embeddings
    seq_df['original_index'] = seq_df.index
    
    # Merge with labels, keeping only samples that have labels
    full_df = seq_df.merge(lbl_df, on='EntryID', how='inner')
    
    # Filter embeddings to match the merged dataframe
    valid_indices = full_df['original_index'].values
    filtered_embeddings = embeddings[valid_indices]
    
    print(f"   Final Train Shape: DF {full_df.shape}, Embeddings {filtered_embeddings.shape}")
    
    # Identify label columns
    exclude_cols = ['EntryID', 'original_index'] 
    label_cols = [c for c in full_df.columns if c in lbl_df.columns and c not in exclude_cols]
    
    labels = full_df[label_cols].values
    
    return filtered_embeddings, labels, label_cols, full_df['EntryID'].values

def main():
    print("--- Starting One-vs-Rest Logistic Regression Training ---")
    
    # 1. Load Data
    X, y, label_columns, entry_ids = load_data()
    
    # 2. Split Data
    print("\n3. Splitting data into Train and Validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train set: {X_train.shape}, {y_train.shape}")
    print(f"   Validation set: {X_val.shape}, {y_val.shape}")
    
    del X, y
    gc.collect()

    # 3. Train Model
    print("\n4. Training One-vs-Rest Logistic Regression model...")
    # Using liblinear as it's good for this scale. C=1.0 is a reasonable default for regularization.
    model = OneVsRestClassifier(
        LogisticRegression(solver='liblinear', C=1.0, random_state=42, tol=1e-3),
        n_jobs=-1 # Use all available cores
    )
    
    model.fit(X_train, y_train)
    print("   Training complete.")
    
    # 4. Save Model
    model_filename = "logistic_regression_c.joblib"
    print(f"\n5. Saving model to {model_filename}...")
    joblib.dump(model, model_filename)
    print("   Model saved.")
    
    # 5. Evaluate on Validation Set
    print("\n6. Evaluating model on validation set...")
    val_preds_proba = model.predict_proba(X_val)
    
    # Calculate F-max
    fmax, precision, recall, best_tau = calculate_fmax(y_val, val_preds_proba)
    
    print(f"\n--- Validation Metrics ---")
    print(f"   F-max: {fmax:.4f}")
    print(f"   Precision @ F-max: {precision:.4f}")
    print(f"   Recall @ F-max: {recall:.4f}")
    print(f"   Best Threshold (tau): {best_tau:.2f}")
    print("--------------------------\n")
    
    # Optional: Predict on test set if it exists
    if os.path.exists(cfg.test_path) and os.path.exists(cfg.test_embedding_path):
        print("7. Generating predictions for the test set...")
        test_df = pd.read_csv(cfg.test_path)
        test_emb = np.load(cfg.test_embedding_path)

        print(f"   Loaded {len(test_df)} test proteins.")
        
        test_preds = model.predict_proba(test_emb)
        
        preds_df = pd.DataFrame(test_preds, columns=label_columns)
        preds_df['EntryID'] = test_df['EntryID']
        
        melted_preds = preds_df.melt(id_vars='EntryID', var_name='term', value_name='score')
        
        # Filter based on the best threshold found during validation
        filtered_submission = melted_preds[melted_preds['score'] > best_tau]
        
        output_filename = f"submission_lr_c_{best_tau:.2f}.tsv"
        print(f"   Saving submission file to {output_filename}...")
        filtered_submission[['EntryID', 'term', 'score']].to_csv(
            output_filename, 
            sep='\\t', 
            index=False, 
            header=False
        )
        print("   Done.")

if __name__ == "__main__":
    main()
