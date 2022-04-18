


path = {
    'data_path': 'data/sst2/',
    # 'electra_path': 'bhadresh-savani/electra-base-emotion',
    # 'electra_path': 'pretrained_model/electra-base-discriminator',
    'electra_path': 'google/electra-small-discriminator',
    'bert_path': 'bert-base-cased'
    # 'electra_path': 'bert-base-cased',
}

cfg = {
    'hidden_dim': 256,
    'dropout': 0.2,
    'batch_size': 16,
    'epoch': 10,
    'learning_rate': 1e-4,
    'electra_flag': False,  # False 代表使用Bert base cased模型
}
