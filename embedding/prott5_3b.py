import os
import re
import gc
import numpy as np
import pandas as pd
import torch
import tqdm
import modal
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
INPUT_FILENAMES = ["sequence.parquet"]
OUTPUT_PREFIX = "t5_embeddings_"
MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"

# --- CẤU HÌNH QUAN TRỌNG ---
BATCH_SIZE = 64             # Có thể tăng lên 16 hoặc 32 vì đã giới hạn độ dài
MAX_SEQUENCE_LENGTH = 1024  # Cắt cụt mọi chuỗi dài hơn 2048 token. 
                            # Giúp tránh OOM triệt để.
SEQ_COLUMN_NAME = "seq"
# ---------------------------

app = modal.App(name="prot-t5-h100-embedder")

prot_t5_image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "pandas",
            "pyarrow",
            "numpy",
            "tqdm",
            "torch==2.3.0",
            "transformers==4.42.3",
            "accelerate",
            "huggingface-hub",
            "sentencepiece",
        ]
    )
)

# ==========================================
# II. HELPER CLASSES
# ==========================================

def clean_sequence_helper(seq_str):
    if pd.isna(seq_str) or seq_str == "":
        return "X"
    seq_str = str(seq_str)
    clean = re.sub(r"[UZOB]", "X", seq_str)
    # ProtT5 cần khoảng trắng giữa các acid amin
    return " ".join(list(clean)) 

class ProteinDataset(Dataset):
    def __init__(self, data_tuples):
        # data_tuples: list of (original_index, sequence_string)
        self.data = data_tuples
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# III. INFERENCE FUNCTION (ON H100)
# ==========================================

@app.function(
    image=prot_t5_image,
    gpu="H100", 
    timeout=7200 * 6
)
def run_inference_on_modal(sequences):
    # 1. Preprocessing
    print(f"Preprocessing {len(sequences)} sequences...")
    processed_seqs = [clean_sequence_helper(s) for s in sequences]
    
    # 2. SMART BATCHING (Vẫn giữ để chạy nhanh hơn)
    # Gắn index gốc
    indexed_data = list(enumerate(processed_seqs)) 
    # Sắp xếp dài -> ngắn
    indexed_data.sort(key=lambda x: len(x[1]), reverse=True)
    
    # 3. DataLoader
    dataset = ProteinDataset(indexed_data)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # 4. Load Model
    print(f"Loading Model (Max Len: {MAX_SEQUENCE_LENGTH})...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    
    device = "cuda"
    model = T5EncoderModel.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    temp_results = []
    
    print(f"Starting inference loop...")
    
    for batch_data in tqdm.tqdm(dataloader):
        batch_indices, batch_texts = batch_data
        
        # === [QUAN TRỌNG]: THÊM TRUNCATION VÀO ĐÂY ===
        ids = tokenizer.batch_encode_plus(
            batch_texts, 
            add_special_tokens=True, 
            padding="longest", 
            return_tensors="pt",
            
            # Cài đặt cắt ngắn
            truncation=True, 
            max_length=MAX_SEQUENCE_LENGTH 
        )
        
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
            
        last_hidden_state = embedding_repr.last_hidden_state

        # Đưa về CPU
        last_hidden_state = last_hidden_state.detach().cpu()
        attention_mask = attention_mask.detach().cpu()

        for j in range(len(batch_texts)):
            # Lấy chiều dài thực tế (sau khi đã truncate)
            seq_len = attention_mask[j].sum()
            
            seq_emb_tensor = last_hidden_state[j, :seq_len]
            
            # Mean Pooling -> Float16 Numpy
            seq_emb = seq_emb_tensor.float().mean(dim=0).numpy().astype(np.float16)
            
            original_idx = batch_indices[j].item()
            temp_results.append((original_idx, seq_emb))
            
        # Clear VRAM
        del ids, input_ids, attention_mask, embedding_repr, last_hidden_state
    
    # 5. Sắp xếp lại đúng thứ tự ban đầu
    print("Sorting results back to original order...")
    temp_results.sort(key=lambda x: x[0])
    final_embeddings = [x[1] for x in temp_results]
    
    return final_embeddings

# ==========================================
# IV. ENTRYPOINT
# ==========================================

@app.local_entrypoint()
def main():
    local_data_dir = "data" 
    os.makedirs(local_data_dir, exist_ok=True)
    
    for input_filename in INPUT_FILENAMES:
        input_path = os.path.join(local_data_dir, input_filename)
        output_path = OUTPUT_PREFIX + input_filename.replace(".parquet", ".npy")
        
        print("="*60)
        print(f"STARTING FILE: {input_filename}")
        
        if not os.path.exists(input_path):
            print(f"SKIP: Missing {input_path}")
            continue
            
        df = pd.read_parquet(input_path)
        sequences = df[SEQ_COLUMN_NAME].tolist()
        print(f"Loaded {len(sequences)} sequences.")

        print("Sending to Modal H100...")
        embeddings_list = run_inference_on_modal.remote(sequences)

        print("Converting to Numpy Float16...")
        embeddings = np.array(embeddings_list, dtype=np.float16)
        
        np.save(output_path, embeddings)
        print(f"SAVED: {output_path} | Shape: {embeddings.shape}")
        
    print("="*60)
    print("ALL DONE.")