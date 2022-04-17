
import torch.nn as nn
from config.cfg import path, cfg
from transformers import ElectraModel, ElectraForSequenceClassification

class ELECTRA(nn.Module):

    def __init__(self):
        super(ELECTRA, self).__init__()
        self.electra = ElectraModel.from_pretrained(path['electra_path'])
        self.dropout = nn.Dropout(cfg['dropout'])
        self.fc = nn.Linear(cfg['hidden_dim'], 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.electra(input_ids, token_type_ids, attention_mask)
        x = x[0]
        x = x[:][0]
        print(x.shape)
        x = self.dropout(x)
        x = self.fc(x)
        return x
