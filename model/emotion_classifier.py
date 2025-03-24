import torch
import torch.nn as nn
from transformers import BertModel

class EmotionClassifier(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", num_classes=28):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output  # [batch_size, hidden_size]
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)      # [batch_size, num_classes]
        return torch.sigmoid(logits)           # Apply sigmoid for multi-label
