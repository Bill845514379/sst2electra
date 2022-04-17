
from datasets import load_dataset
from config.cfg import cfg, path
from transformers import ElectraTokenizer, ElectraModel
import numpy as np
import torch

tokenizer = ElectraTokenizer.from_pretrained(path['electra_path'], do_lower_case=True)

def tran_list(data):
  t = []
  for i in range(len(data)):
    t.append(data[i])
  return t

def load_data(path):
    from datasets import load_from_disk
    sst2 = load_from_disk(path)
    sst2.set_format("pandas")

    data_train = sst2['train']
    data_val = sst2['validation']
    data_test = sst2['test']
    print(data_train['sentence'][0])
    print(data_train['tokens'][0])
    x_train = tokenizer(tran_list(data_train['sentence']), padding=True, return_tensors="pt")
    y_train = torch.tensor(data_train['label'])

    x_val = tokenizer(tran_list(data_val['sentence']),padding=True, return_tensors="pt")
    y_val = torch.tensor(data_val['label'])

    x_test = tokenizer(tran_list(data_test['sentence']) ,padding=True, return_tensors="pt")
    y_test = torch.tensor(data_test['label'])

    return x_train, y_train, x_val, y_val, x_test, y_test



