
from transformers import ElectraForSequenceClassification, ElectraConfig
import numpy as np
from common.load_data import load_data
from common.set_random_seed import setup_seed
from torch.autograd import Variable
from config.cfg import path, cfg
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import time

seeds = [44, 105,1024, 422, 752]
average_acc = 0
for ss in seeds:
    setup_seed(ss)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('可用设备：', device)

    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data(path['data_path'])
    train_data = TensorDataset(X_train['input_ids'], X_train['token_type_ids'], X_train['attention_mask'], y_train)
    dev_data = TensorDataset(X_dev['input_ids'], X_dev['token_type_ids'], X_dev['attention_mask'], y_dev)
    test_data = TensorDataset(X_test['input_ids'], X_test['token_type_ids'], X_test['attention_mask'], y_test)

    loader_train = DataLoader(
        dataset=train_data,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    loader_dev = DataLoader(
        dataset=dev_data,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    loader_test = DataLoader(
        dataset=test_data,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    configuration = ElectraConfig.from_pretrained(path['electra_path'])
    configuration['num_labels'] = 2
    net = ElectraForSequenceClassification.from_pretrained(path['electra_path'], config=configuration)
    net.to(device)

    epoch = cfg['epoch']
    optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    print(cfg)

    for i in range(epoch):
        print('-------------------------   training   ------------------------------')
        time0 = time.time()
        batch = 0
        ave_loss, sum_acc = 0, 0

        for batch_x, token_type_ids, attention_mask, batch_y in loader_train:
            net.train()
            batch_x, token_type_ids, attention_mask, batch_y = Variable(batch_x).long(), Variable(
                token_type_ids).long(), Variable(attention_mask).long(), Variable(batch_y).long()
            batch_x, token_type_ids, attention_mask, batch_y = batch_x.to(device), token_type_ids.to(
                device), attention_mask.to(device), batch_y.to(device)

            output = net(input_ids = batch_x, token_type_ids = token_type_ids, attention_mask=attention_mask).logits
            criterion = nn.CrossEntropyLoss()
            # print(output.shape)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()  # 更新权重
            ave_loss += loss
            batch += 1

            if batch % 200 == 0:
                print('epoch:{}/{},batch:{}/{},time:{}, loss:{},learning_rate:{}'.format(i + 1, epoch, batch,
                                                                                         len(loader_train),
                                                                                         round(time.time() - time0, 4),
                                                                                         loss,
                                                                                         optimizer.param_groups[
                                                                                             0]['lr']))
        scheduler.step(loss)

        print('------------------ epoch:{} ----------------'.format(i + 1))
        print('train_average_loss{}'.format(ave_loss / len(loader_train)))
        print('============================================'.format(i + 1))

        time0 = time.time()
        if (i + 1) % 2 == 0:
            label_out, label_y = [], []
            print('-------------------------   dev   ------------------------------')
            sum_acc, num = 0, 0
            # torch.save(net.state_dict(), 'save_model/params' + str(i + 1) + '.pkl')
            for batch_x, token_type_ids, attention_mask, batch_y in loader_dev:
                net.eval()
                batch_x, token_type_ids, attention_mask, batch_y = Variable(batch_x).long(), Variable(
                    token_type_ids).long(), Variable(attention_mask).long(), Variable(batch_y).long()
                batch_x, token_type_ids, attention_mask, batch_y = batch_x.to(device), token_type_ids.to(
                    device), attention_mask.to(device), batch_y.to(device)

                with torch.no_grad():
                     output = net(input_ids=batch_x, token_type_ids=token_type_ids, attention_mask=attention_mask).logits

                _, pred = torch.max(output, dim=1)

                pred = pred.cpu().detach().numpy()
                batch_y = batch_y.cpu().detach().numpy()

                for j in range(pred.shape[0]):
                    label_out.append(pred[j])
                    label_y.append(batch_y[j])

            label_out = np.array(label_out)
            label_y = np.array(label_y)

            acc = (np.sum(label_y == label_out)) / len(label_y)
            print('------------------ epoch:{} ----------------'.format(i + 1))
            print('dev_acc:{}, time:{}'.format( round(acc, 4), time.time()-time0))
            print('============================================'.format(i + 1))

        if (i + 1) % epoch == 0:
            label_out, label_y = [], []
            print('-------------------------   dev   ------------------------------')
            sum_acc, num = 0, 0
            for batch_x, token_type_ids, attention_mask, batch_y in loader_test:
                net.eval()
                batch_x, token_type_ids, attention_mask, batch_y = Variable(batch_x).long(), Variable(
                    token_type_ids).long(), Variable(attention_mask).long(), Variable(batch_y).long()
                batch_x, token_type_ids, attention_mask, batch_y = batch_x.to(device), token_type_ids.to(
                    device), attention_mask.to(device), batch_y.to(device)

                with torch.no_grad():
                     output = net(input_ids=batch_x, token_type_ids=token_type_ids, attention_mask=attention_mask).logits

                _, pred = torch.max(output, dim=1)

                pred = pred.cpu().detach().numpy()
                batch_y = batch_y.cpu().detach().numpy()

                for j in range(pred.shape[0]):
                    label_out.append(pred[j])
                    label_y.append(batch_y[j])

            label_out = np.array(label_out)
            label_y = np.array(label_y)

            acc = (np.sum(label_y == label_out)) / len(label_y)
            average_acc += acc
            print('------------------ epoch:{} ----------------'.format(i + 1))
            print('test_acc:{}, time:{}'.format( round(acc, 4), time.time()-time0))
            print('============================================'.format(i + 1))

print('attention:', cfg['attention'], average_acc/len(seeds))
# torch.save(net.state_dict(), './save_model/params.pkl')

