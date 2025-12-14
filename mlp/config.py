from types import SimpleNamespace

cfg = SimpleNamespace( 
    learning_rate = 1e-3,
    batch_size = 64,    
    num_workers = 4,
    input_dim = 1024,      
    taxonomy_dim = 64,     
    hidden_dim = 512,      
    dropout = 0.1,
    experiment_name = "CAFA6_MultiModal_v3",
    tracking_uri = None,
    num_epochs = 100,     
    patience = 2,
    loss = 'bce', 
    gamma = 2.0,
    sequence_path = "sequence.parquet",  #hfuehfueh
    train_taxonomy_path = "taxon.csv", 
    embedding_path = "t5_embeddings_sequence.npy", 
    label_path = "label_p.parquet", #cco, bpo, mfo
    ia_path = "ias.csv",
    test_path = "test_protein.csv",
    test_taxonomy_path = "taxon.csv",
    test_embedding_path = "t5_embeddings_res_test.npy", # Sửa lại đường dẫn nếu cần
    top_k = 1000, # Giữ lại top_k nhãn phổ biến nhất
    output = "submission_mlp_p.parquet"
)