from os import path
import torch
from torch.utils.data import Dataset, DataLoader
from underthesea import word_tokenize
from mint.config import DATA_DIR


def load_uit_vsfc(subset='train'):
    assert subset in ['train', 'test', 'val'], "Subset must be 'train' or 'test' or 'val'"
    
    UIT_VSFC_DIR = path.join(DATA_DIR, "UIT-VSFC")
    data_path = path.join(UIT_VSFC_DIR, subset, 'sents.txt')
    labels_path = path.join(UIT_VSFC_DIR, subset, 'sentiments.txt')

    # Load the data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    data = [line.strip() for line in data]

    # Load the labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [int(line.strip()) for line in labels]

    if len(data) != len(labels):
        raise ValueError("Data and labels must have the same length")
    
    return data, labels


class VSFCDataset(Dataset):
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
    def __init__(self, tokenizer, batch_size=32, max_length=128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def load_data(self, subset='train'):
        texts, labels = load_uit_vsfc(subset)
        texts = [word_tokenize(text, format='text') for text in texts]
        dataset = VSFCDataset(texts, labels, self.tokenizer, self.max_length)

        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=(subset == 'train'), 
            num_workers=4, 
            pin_memory=True
        )
