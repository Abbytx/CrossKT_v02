import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import sys
import numpy as np
sys.path.extend(['../'])
import time
import argparse

from sklearn import metrics
from Models import CrossKTnet
from Dataset import adni2, adni3


def cacu_metric(output, y):
    predict = torch.argmax(output, dim=-1)
    ACC = torch.sum(predict == y)
    add = (y + predict).cpu()
    sub = (y - predict).cpu()
    TP = str(add.numpy().tolist()).count('2.0')
    TN = str(add.numpy().tolist()).count('0.0')
    FP = str(sub.numpy().tolist()).count('-1')
    FN=len(y)-TP-TN-FP
    return ACC / len(y), TP, TN, FP, FN, TP / (TP + FN), TN / (TN + FP)



def print_log(string, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        string = "[ " + localtime + ' ] ' + string
    with open('{}/{}_train_log.txt'.format(args.work_dir, args.dataset), 'a') as f:
        print(string, file=f)


def log_configuration(args, num_train, num_valid, configs):
    print_log('------------------GPU initialization！ -----------------------------')

    print_log('CUDA_VISIBLE_DEVICES:{}'.format(args.gpu))
    print_log('****************dataset details********************')
    print_log('dataset:{}'.format(args.dataset))
    print_log('Samples for train = {}  Samples for valid = {} '.format(num_train, num_valid))

    print_log('*****************train setting********************')
    print_log('train epoch={}  '.format(args.end_epoch))
    print_log('batch_size={}  '.format(args.batch_size))
    print_log('lr={} momentum={}'.format(args.lr, args.momentum))

    print_log('***************model details********************')
    print_log('sparsity_alpha:{}'.format(args.sparsity_alpha))
    print_log('kernel_size:{}'.format(args.kernel_size))
    print_log('config:{}'.format(configs))


def data_selection(dataset, split):
    if  dataset == 'adni2':
        train_data = adni2(split=split, mode='train')
        valid_data = adni2(split=split, mode='test')
        one_sample, ___ = train_data.__getitem__(1)
        num_frame = one_sample.shape[-3]
        num_point = one_sample.shape[-2]
        num_class = valid_data.get_num_class()

    elif dataset == 'adni3':
        train_data = adni3(split=split, mode='train')
        valid_data = adni3(split=split, mode='test')

        one_sample, ___ = train_data.__getitem__(1)
        num_frame = one_sample.shape[-3]
        num_point = one_sample.shape[-2]

        num_class = valid_data.get_num_class()

    return train_data, valid_data, num_frame, num_point, num_class,


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        '''random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True'''

    # Save the optimal values of 5-folds
    all_best_ACC = np.zeros(5)
    all_best_SEN = np.zeros(5)
    all_best_SPE = np.zeros(5)
    all_best_AUC = np.zeros(5)

    for k in args.k_folds:
        train_data, valid_data, num_frame, num_point, num_class = data_selection(args.dataset, split=k)
        num_train = len(train_data)
        num_valid = len(valid_data)

        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True)  # worker_init_fn=np.random.seed(args.seed)

        valid_loader = DataLoader(dataset=valid_data, batch_size=num_valid, shuffle=False,
                                  drop_last=False, pin_memory=True)

        torch.cuda.set_device(args.gpu)
        if os.path.exists("./check_points") is False:
            os.makedirs('./check_points')  # Save weights

        log_configuration(args, num_train, num_valid, args.config_128)

        print_log('num_frame={} num_point={}'.format(num_frame, num_point))
        print_log('{}'.format(args.root))
        print_log('---------------------{}split------------------'.format(k))


        My_mode = CrossKTnet(sparsity_alpha=args.sparsity_alpha, num_subset=args.num_subset, num_frame=num_frame, num_point=num_point,
                             kernel_size=args.kernel_size, use_pes=args.use_pes,num_class=num_class,config=args.config)


        My_mode = My_mode.cuda(args.device)
        My_mode = torch.nn.DataParallel(My_mode, device_ids=[args.gpu])

        if args.pre_trained:  # Load the pre-training weight
            print_log('loading   weights：{}'.format(args.weights_path))
            weights_dict = torch.load(args.weights_path, map_location=lambda storage, loc: storage)
            My_mode.load_state_dict(weights_dict['state_dict'])



        optimizer = torch.optim.SGD(My_mode.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True,
                                    weight_decay=0.0001)
        loss_F = nn.CrossEntropyLoss().to(args.device)

        Best_ACC= 0

        for epoch in range(args.start_epoch, args.end_epoch):
            My_mode.train()  # Train
            if epoch < args.warm_up_epoch:
                lr = args.lr * (epoch + 1) / args.warm_up_epoch
            else:
                lr = args.lr * (args.lr_decay_rate ** np.sum(epoch >= np.array(args.step)))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            total_loss = 0
            train_ACC = 0
            for i, data in enumerate(train_loader):
                x, target = data
                x = x.cuda(args.device, non_blocking=True)
                target = target.cuda(args.device, non_blocking=True)
                output = My_mode(x)

                loss = loss_F(output, target)
                total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_ACC += torch.sum((torch.argmax(output, dim=-1) == target))


            My_mode.eval() # Test
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    x, target = data
                    x = x.cuda(args.device, non_blocking=True)
                    target = target.cuda(args.device, non_blocking=True)
                    out_put = My_mode(x)

                ACC, TP, TN, FP, FN, sen, spe, auc = cacu_metric(out_put=out_put, y=target)

            if ACC > Best_ACC:
                Best_ACC= ACC
                all_best_SEN[k - 1] = sen
                all_best_SPE[k - 1] =spe
                all_best_AUC[k - 1] = auc

            print_log( 'split:{} Epoch: {}  loss:{:.5f}  train_ACC:{:.5f} test_ACC:{:.5f} Best_ACC:{:.5f} '.format(k,
                                                                    epoch, total_loss / len(train_loader),train_ACC / num_train,ACC, Best_ACC))

            print_log('TP:{}  TN:{}  FP:{} FN:{} '.format(TP, TN, FP, FN))
            print_log( 'SEN: {:.5f}  SPE: {:.5f} auc:{:.5f}'.format(sen,spe, auc))
            all_best_ACC[k - 1] = Best_ACC

            if ((epoch + 1) % int(args.save_freq) == 0):
                file_name = os.path.join('./check_points/{}_split{}_epoch_{}.pth'.format(args.dataset, k, epoch))  # checkpoint_dir
                torch.save({
                    'epoch': epoch,
                    'state_dict': My_mode.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

    with open('final_results.txt', 'a') as f:
        localtime = time.asctime(time.localtime(time.time()))

        print(args.dataset, file=f)
        print("[ " + localtime + ' ] ' + 'SEN: {} Average: {:5.f} std: {:5.f}'.format(all_best_SEN, np.mean(all_best_SEN),
                                                                             (np.std(all_best_SEN))), file=f)
        print("[ " + localtime + ' ] ' + 'SPE: {} Average: {:5.f} std: {:5.f}'.format(all_best_SPE, np.mean(all_best_SPE),
                                                                             (np.std(all_best_SPE))), file=f)
        print("[ " + localtime + ' ] ' + 'AUC: {} Average: {:5.f} std: {:5.f}'.format(all_best_AUC, np.mean(all_best_AUC),
                                                                             (np.std(all_best_AUC))), file=f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # initialize the parameters of the model

    parser.add_argument('--all_datasets', default=['adni2', 'adni3'])  # 'amci_nci','namci_nci'  ,'amci_nci','namci_nci'
    parser.add_argument('--dataset', default='adni2', type=str)  # amci_nci  namci_nci amci_namci  amci_namci_nci
    parser.add_argument('--k_folds', default=[1,2,3,4,5])

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)


    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--pre_trained', default=False, type=str)
    parser.add_argument('--weights_path', default='path.....', type=str)

    parser.add_argument('--work_dir', default='./logging')

    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=250, type=int)
    parser.add_argument('--save_freq', default=1, type=int)  # 保存模型的频率
    parser.add_argument('--step', default=[220, 230])
    parser.add_argument('--warm_up_epoch', default=5, type=int)


    parser.add_argument('--config_128', default= [[6, 12, False, 1, 32,7, 2],   # in_channels, out_channels, is_regularization, Conv_stride,chunk_kernel_size,total_chunk,num_subset
                                                 [12, 12, True, 2, 1, 1, 2],
                                                 [12, 12, False, 1, 32, 3, 2],
                                                 [12, 12, True, 2, 1, 1, 2],
                                                 [12, 12, True, 1, 1, 1, 2],])

    # model parameters:
    #parser.add_argument('--use_kernel_attention', default=1, type=int)  # [0,1,2]
    parser.add_argument('--sparsity_alpha', default=2, type=float)  # 0,[1~2]
    parser.add_argument('--kernel_size', default=1, type=int)
    parser.add_argument('--use_pes', default=True)

    args = parser.parse_args()
    for each in args.all_datasets:
        args.dataset = each
        main(args)








