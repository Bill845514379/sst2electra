


path = {
    'data_path': 'data/sst2/',
    'electra_path': 'pretrained_model/electra-base-discriminator'
}

cfg = {
    'batch_size': 1,
    'epoch': 10,
    'learning_rate': 1e-4,
}

hyper = {
    'lstm_hidden': 256,
    'word_dim': 300,
    'dropout': 0.5
}