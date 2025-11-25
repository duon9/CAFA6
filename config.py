from types import SimpleNamespace

cfg = SimpleNamespace( 
    learning_rate = 1e-4,
    batch_size = 64,    
    num_workers = 4,
    input_dim = 1024,      
    taxonomy_dim = 32,     
    hidden_dim = 512,      
    dropout = 0.1,
    experiment_name = "CAFA6_MultiModal_v3",
    tracking_uri = "http://112.137.129.161:8836",
    num_epochs = 100,     
    patience = 2,
    loss = 'weighted_focal', 
    gamma = 2.0,
    sequence_path = "kaggle/input/sequence.parquet", 
    train_taxonomy_path = "kaggle/input/taxon.csv", 
    embedding_path = "kaggle/input/t5_embeddings_sequence.npy", 
    label_path = "kaggle/input/label_c.parquet",
    ia_path = "kaggle/input/ias.csv",
    test_path = "kaggle/input/test_protein.csv",
    test_taxonomy_path = "kaggle/input/taxon.csv",
    test_embedding_path = "kaggle/input/t5_embeddings_test.npy",
    top_k = 100,
    output = "submissionc.tsv"
)