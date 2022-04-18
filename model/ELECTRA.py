import torch
import torch.nn as nn
from config.cfg import path, cfg
from transformers import ElectraModel, ElectraForSequenceClassification, BertModel

class ELECTRA(nn.Module):

    def __init__(self):
        super(ELECTRA, self).__init__()
        if cfg['electra_flag']:
            self.model = ElectraModel.from_pretrained(path['electra_path'])
        else:
            self.model = BertModel.from_pretrained(path['bert_path'])

        self.dropout = nn.Dropout(cfg['dropout'])
        self.fc = nn.Linear(cfg['hidden_dim'], 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = x[0]
        x = x[:, 0, :]
        # x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
