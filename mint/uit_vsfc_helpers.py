from os import path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from underthesea import word_tokenize
from mint.config import DATA_DIR


def load_uit_vsfc(subset='train'):
    """
    Load the UIT-VSFC dataset.
    Supports 'train', 'test', and 'val' splits.
    """
    assert subset in ['train', 'test', 'val'], "Subset must be 'train', 'test' or 'val'"
    base = path.join(DATA_DIR, "UIT-VSFC")
    data_path = path.join(base, subset, 'sents.txt')
    labels_path = path.join(base, subset, 'sentiments.txt')

    with open(data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f]

    if len(texts) != len(labels):
        raise ValueError("UIT-VSFC: data and labels must have the same length")
    return texts, labels


def load_aivivn(subset='train'):
    """
    Load the AIVIVN-2019 dataset.
    Supports 'train' and 'test' splits only.
    Assumes CSV has columns for text and label.
    """
    assert subset in ['train', 'test'], "AIVIVN only supports 'train' and 'test'"
    csv_path = path.join(DATA_DIR, "AIVIVN-2019", f"{subset}.csv")
    df = pd.read_csv(csv_path)

    # Detect columns
    if 'text' in df.columns and 'label' in df.columns:
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
    elif 'sentence' in df.columns and 'sentiment' in df.columns:
        texts = df['sentence'].astype(str).tolist()
        labels = df['sentiment'].astype(int).tolist()
    else:
        # fallback: assume second column is text, third is label
        texts = df.iloc[:, 1].astype(str).tolist()
        labels = df.iloc[:, 2].astype(int).tolist()

    if len(texts) != len(labels):
        raise ValueError("AIVIVN: data and labels must have the same length")
    return texts, labels


class VSFCDataset(Dataset):
    """
    PyTorch Dataset for UIT-VSFC sentence-level sentiment.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class AIVIVNDataset(Dataset):
    """
    PyTorch Dataset for AIVIVN-2019 sentence-level sentiment.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class VSFCLoader:
    """
    DataLoader factory for UIT-VSFC dataset.
    """
    def __init__(self, tokenizer, batch_size=32, max_length=128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def load_data(self, subset='train'):
        texts, labels = load_uit_vsfc(subset)
        texts = [word_tokenize(t, format='text') for t in texts]
        ds = VSFCDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=(subset=='train'), num_workers=4, pin_memory=True)


class AIVIVNLoader:
    """
    DataLoader factory for AIVIVN-2019 dataset.
    """
    def __init__(self, tokenizer, batch_size=32, max_length=128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def load_data(self, subset='train'):
        texts, labels = load_aivivn(subset)
        texts = [word_tokenize(t, format='text') for t in texts]
        ds = AIVIVNDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=(subset=='train'), num_workers=4, pin_memory=True)

