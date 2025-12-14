import os
import re
import gc
import numpy as np
import pandas as pd
import torch
import tqdm
import modal
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
INPUT_FILENAMES = ["res_test.parquet"]
OUTPUT_PREFIX = "esm2_3b_embeddings_"

# MODEL ESM-2 3B
MODEL_NAME = "facebook/esm2_t36_3B_UR50D"

BATCH_SIZE = 128             
MAX_SEQUENCE_LENGTH = 1024  
SEQ_COLUMN_NAME = "seq"

app = modal.App(name="esm2-3b-h100-embedder")

esm2_image = (
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
    # ESM-2 chấp nhận aa chuẩn, thay U Z O B → X
    clean = re.sub(r"[UZOB]", "X", seq_str)
    return clean  # KHÔNG thêm khoảng trắng như ProtT5

class ProteinDataset(Dataset):
    def __init__(self, data_tuples):
        self.data = data_tuples
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# III. INFERENCE FUNCTION (ON H100)
# ==========================================

@app.function(
    image=esm2_image,
    gpu="H100", 
    timeout=7200 * 6
)
def run_inference_on_modal(sequences):
    print(f"Preprocessing {len(sequences)} sequences...")
    processed_seqs = [clean_sequence_helper(s) for s in sequences]
    
    indexed_data = list(enumerate(processed_seqs)) 
    indexed_data.sort(key=lambda x: len(x[1]), reverse=True)
    
    dataset = ProteinDataset(indexed_data)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    print("Loading ESM2 3B model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    device = "cuda"
    model = EsmModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    temp_results = []
    print("Starting inference...")

    for batch_data in tqdm.tqdm(dataloader):
        batch_indices, batch_texts = batch_data
        
        ids = tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        )
        
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state.detach().cpu()
        attention_mask = attention_mask.detach().cpu()

        for j in range(len(batch_texts)):
            seq_len = attention_mask[j].sum()

            emb_tensor = last_hidden_state[j, :seq_len]

            seq_emb = emb_tensor.float().mean(dim=0).numpy().astype(np.float16)

            original_idx = batch_indices[j].item()
            temp_results.append((original_idx, seq_emb))

        del ids, input_ids, attention_mask, outputs, last_hidden_state

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

        print("Converting to NumPy...")
        embeddings = np.array(embeddings_list, dtype=np.float16)
        
        np.save(output_path, embeddings)
        print(f"SAVED: {output_path} | Shape: {embeddings.shape}")
        
    print("="*60)
    print("ALL DONE.")
