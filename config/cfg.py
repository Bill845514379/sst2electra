


path = {
    'data_path': 'data/sst2/',
    'electra_path': 'google/electra-base-discriminator'
}

cfg = {
    'batch_size': 16,
    'epoch': 10,
    'learning_rate': 1e-5,
}

hyper = {
    'lstm_hidden': 256,
    'word_dim': 300,
    'dropout': 0.5
}
