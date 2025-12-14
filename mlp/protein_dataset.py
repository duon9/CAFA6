import pandas as pd
import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader

class ProteinEmbeddingDataset(Dataset):
    def __init__(self,
                 embeddings: np.ndarray,
                 dataframe: pd.DataFrame,
                 label_columns: list = None,
                 taxonomy_map: dict = None):
        
        self.embeddings = embeddings
        self.dataframe = dataframe
        self.label_columns = label_columns
        self.taxonomy_map = taxonomy_map
        
        # Tự động tìm cột ID
        if 'EntryID' in self.dataframe.columns:
            self.id_col = 'EntryID'
        elif 'id' in self.dataframe.columns:
            self.id_col = 'id'
        else:
            self.id_col = self.dataframe.columns[0]
            
        # Kiểm tra cột Taxonomy
        self.tax_col = 'taxonomyID'
        # Default index cho unknown tax (thường là 0)
        self.unknown_tax_idx = 0 
        if taxonomy_map is not None:
             self.unknown_tax_idx = taxonomy_map.get('<unknown>', 0)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        # 1. Sequence Embedding
        emb_vector = self.embeddings[index]
        data_row = self.dataframe.iloc[index]
        protein_id = data_row[self.id_col]
        
        item = dict(
            protein_id=protein_id, 
            embedding=torch.tensor(emb_vector, dtype=torch.float32)
        )
        
        # 2. Taxonomy Indexing
        if self.taxonomy_map is not None and self.tax_col in data_row:
            raw_tax = data_row[self.tax_col]
            # Map raw tax ID sang index, nếu không có trong map thì về unknown
            tax_idx = self.taxonomy_map.get(raw_tax, self.unknown_tax_idx)
            item['taxonomy_idx'] = torch.tensor(tax_idx, dtype=torch.long)
        else:
            # Fallback an toàn
            item['taxonomy_idx'] = torch.tensor(self.unknown_tax_idx, dtype=torch.long)
        
        # 3. Labels
        if self.label_columns is not None:
            labels = data_row[self.label_columns].values.astype(float)
            item['labels'] = torch.FloatTensor(labels)
            
        return item


class LitProteinDataModule(L.LightningDataModule):
    def __init__(self,
                 train_df: pd.DataFrame,
                 train_emb: np.ndarray,
                 val_df: pd.DataFrame,
                 val_emb: np.ndarray,
                 label_columns: list,
                 taxonomy_map: dict,
                 test_df: pd.DataFrame = None,
                 test_emb: np.ndarray = None,
                 batch_size: int = 32,
                 num_workers: int = 4):
        super().__init__()
        self.train_df = train_df
        self.train_emb = train_emb
        self.val_df = val_df
        self.val_emb = val_emb
        
        self.test_df = test_df
        self.test_emb = test_emb
        
        self.label_columns = label_columns
        self.taxonomy_map = taxonomy_map
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: str = None):
        self.train_dataset = ProteinEmbeddingDataset(
            embeddings=self.train_emb,
            dataframe=self.train_df, 
            label_columns=self.label_columns,
            taxonomy_map=self.taxonomy_map
        )
        
        self.val_dataset = ProteinEmbeddingDataset(
            embeddings=self.val_emb,
            dataframe=self.val_df, 
            label_columns=self.label_columns,
            taxonomy_map=self.taxonomy_map
        )
        
        if self.test_df is not None:
            self.test_dataset = ProteinEmbeddingDataset(
                embeddings=self.test_emb,
                dataframe=self.test_df, 
                label_columns=None,
                taxonomy_map=self.taxonomy_map
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.test_dataset is None: return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)