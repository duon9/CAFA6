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
OUTPUT_PREFIX = "ankh3_xl_embeddings_"

MODEL_NAME = "ElnaggarLab/ankh3-xl"

MAX_TOKEN_PER_CHUNK = 2048   # tối đa token mỗi chunk
BATCH_SIZE_SHORT = 64        # batch size lớn cho sequence ngắn
BATCH_SIZE_LONG = 4          # batch size nhỏ cho sequence dài

SEQ_COLUMN_NAME = "seq"

app = modal.App(name="ankh3-xl-h100-embedder")

ankh3_image = (
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
# HELPER FUNCTIONS
# ==========================================
def clean_sequence_helper(seq_str):
    if pd.isna(seq_str) or seq_str == "":
        return "X"
    seq_str = str(seq_str)
    clean = re.sub(r"[UZOB]", "X", seq_str)
    return clean

class ProteinDataset(Dataset):
    def __init__(self, data_tuples):
        self.data = data_tuples
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# INFERENCE FUNCTION (ON H100)
# ==========================================
@app.function(
    image=ankh3_image,
    gpu="H100",
    timeout=7200 * 6
)
def run_inference_on_modal(sequences):
    print(f"Preprocessing {len(sequences)} sequences...")

    processed_seqs = ["[NLU]" + clean_sequence_helper(s) for s in sequences]

    # Gắn index để sắp xếp lại sau
    indexed_data = list(enumerate(processed_seqs))

    # Tách sequence ngắn vs dài
    short_seqs = [(i, s) for i, s in indexed_data if len(s) < MAX_TOKEN_PER_CHUNK]
    long_seqs = [(i, s) for i, s in indexed_data if len(s) >= MAX_TOKEN_PER_CHUNK]

    device = "cuda"

    print("Loading Ankh3-XL model...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5EncoderModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()

    temp_results = []

    # =======================
    # SHORT SEQUENCES (BATCHED)
    # =======================
    if short_seqs:
        print(f"Processing {len(short_seqs)} short sequences with batch size {BATCH_SIZE_SHORT}...")
        short_dataset = ProteinDataset(short_seqs)
        short_loader = DataLoader(short_dataset, batch_size=BATCH_SIZE_SHORT, shuffle=False)

        for batch_data in tqdm.tqdm(short_loader):
            batch_indices, batch_texts = batch_data

            encoding = tokenizer(
                list(batch_texts),
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN_PER_CHUNK
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            hidden = outputs.last_hidden_state.detach().cpu()
            mask = attention_mask.detach().cpu()

            for j in range(hidden.shape[0]):
                seq_len_masked = mask[j].sum()
                emb = hidden[j, :seq_len_masked].float().mean(dim=0).numpy().astype(np.float16)
                temp_results.append((batch_indices[j].item(), emb))

            del input_ids, attention_mask, outputs, hidden, encoding
            torch.cuda.empty_cache()
            gc.collect()

    # =======================
    # LONG SEQUENCES (CHUNKED)
    # =======================
    if long_seqs:
        print(f"Processing {len(long_seqs)} long sequences (chunked)...")
        for idx, text in tqdm.tqdm(long_seqs):
            seq_len = len(text)
            chunk_embeddings = []

            for start in range(0, seq_len, MAX_TOKEN_PER_CHUNK):
                chunk = text[start:start+MAX_TOKEN_PER_CHUNK]

                ids = tokenizer(
                    chunk,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                input_ids = ids['input_ids'].to(device)
                attention_mask = ids['attention_mask'].to(device)

                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                hidden = outputs.last_hidden_state.detach().cpu()
                mask_chunk = attention_mask.detach().cpu()
                chunk_len_masked = mask_chunk[0].sum()
                emb = hidden[0, :chunk_len_masked].float().mean(dim=0).numpy()
                chunk_embeddings.append(emb)

                del ids, input_ids, attention_mask, outputs, hidden
                torch.cuda.empty_cache()
                gc.collect()

            seq_emb = np.mean(np.stack(chunk_embeddings), axis=0).astype(np.float16)
            temp_results.append((idx, seq_emb))

    # Sắp xếp theo index ban đầu
    temp_results.sort(key=lambda x: x[0])
    final_embeddings = [x[1] for x in temp_results]

    return final_embeddings

# ==========================================
# ENTRYPOINT
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
