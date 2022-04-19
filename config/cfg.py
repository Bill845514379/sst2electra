


path = {
    'data_path': 'data/sst2/',
    # 'electra_path': 'bhadresh-savani/electra-base-emotion',
    # 'electra_path': 'pretrained_model/electra-base-discriminator',
    'electra_path': 'google/electra-base-discriminator',
    'bert_path': 'bert-base-uncased'
}

cfg = {
    'hidden_dim': 768,
    'dropout': 0.2,
    'batch_size': 16,
    'epoch': 20,
    'learning_rate': 1e-5,
    'electra_flag': True,  # False 代表使用Bert base uncased
}
