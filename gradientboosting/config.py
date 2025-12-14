from types import SimpleNamespace

# Cấu hình cho Py-Boost
cfg = SimpleNamespace(
    pyboost_loss='bce',
    ntrees=2000,              
    lr=0.03,                  
    es=150,                    
    max_depth=7,               
    lambda_l2=2,
    use_hess=False,            
    sketch_type='random_projection',
    sketch_dim=32,            
    model_save_path='checkpoints/pyboost_model.pkl', # Đường dẫn lưu model
    sequence_path="sequence.parquet",
    embedding_path="esm2_3b_embeddings_sequence.npy",
    label_path="label_p.parquet",
    ia_path="ias.csv",
    test_path="test_protein.csv",
    test_embedding_path="esm2_3b_embeddings_res_test.npy",
    prediction_threshold=0.01,
    output="submission_pyboost_p_3000.parquet",
    experiment_name="CAFA6_PyBoost_v1",
    tracking_uri="",
    input_dim=1024,
    num_workers=4,
    top_k = 3000,
)
