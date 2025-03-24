import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils.label_map import emotion_labels

class GoEmotionsDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=128):
        self.data = pd.read_csv(filepath)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_labels = len(emotion_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.loc[index, 'text']
        labels = eval(self.data.loc[index, 'labels'])  # labels are stored as list of indices in string form

        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Create multi-hot label vector
        label_vector = torch.zeros(self.num_labels)
        label_vector[labels] = 1.0

        return {
            'input_ids': encoding['input_ids'].squeeze(),        # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_vector
        }

def get_dataloaders(tokenizer, batch_size=16):
    train_dataset = GoEmotionsDataset("data/goemotions_train.csv", tokenizer)
    val_dataset = GoEmotionsDataset("data/goemotions_val.csv", tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
