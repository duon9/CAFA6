from types import SimpleNamespace

# Cấu hình cho Py-Boost
cfg = SimpleNamespace(
    # --- Py-Boost Parameters ---
    pyboost_loss='crossentropy', # Hàm loss cho multi-label classification
    ntrees=2000,               # Số lượng cây tối đa
    lr=0.03,                   # Tốc độ học (learning rate)
    es=150,                    # Early stopping rounds
    max_depth=7,               # Độ sâu tối đa của cây
    lambda_l2=2,               # L2 regularization
    use_hess=False,            # Bắt buộc False khi dùng RandomProjectionSketch
    sketch_type='random_projection',
    sketch_dim=32,             # Giảm chiều gradient từ ~3000 xuống 32 để tăng tốc
    model_save_path='checkpoints/pyboost_model.pkl', # Đường dẫn lưu model

    # --- Data Paths (Giữ nguyên từ config cũ) ---
    sequence_path="kaggle/input/sequence.parquet",
    embedding_path="kaggle/input/t5_embeddings_sequence.npy",
    label_path="kaggle/input/label_c.parquet",
    ia_path="kaggle/input/ias.csv",
    test_path="kaggle/input/test_protein.csv",
    test_embedding_path="kaggle/input/t5_embeddings_res_test.npy",
    
    # --- Prediction & Submission ---
    prediction_threshold=0.15, # Ngưỡng để quyết định nhãn dương tính
    output="submission_pyboost.tsv",
    
    # --- Experiment Tracking (Optional) ---
    experiment_name="CAFA6_PyBoost_v1",
    tracking_uri="http://112.137.129.161:8836",
    
    # --- Unused Parameters (giữ lại để tham khảo) ---
    input_dim=1024,
    num_workers=4,
)
