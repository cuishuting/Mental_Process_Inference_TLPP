import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import transformer.Constants as Constants
import Utils


from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = opt.num_types ##### total number of predicates
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types


def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_loss = 0  # cumulative mental log-likelihood
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        
        mental_time, mental_type, action_time, action_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        ##### 这里只使用action event来做embedding
        enc_out, haz_out = model(action_time, action_type)

        """ backward """
        # negative log-likelihood for mental event
        log_likelihood_mental = Utils.log_likelihood_mental(mental_time, mental_type, haz_out)
        loss = -log_likelihood_mental
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_loss += loss.item()
        
    return -total_loss/len(training_data)
             

def train(model, training_data, testing_data, optimizer, scheduler, opt):
    """ Start training. """
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        log_likelihood_mental = train_epoch(model, training_data, optimizer, opt)
        print('  - (Training)    log_likelihood_mental: {log_likelihood_mental: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(log_likelihood_mental=log_likelihood_mental, elapse=(time.time() - start) / 60))

        ##### logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {log_likelihood_mental: 8.5f}\n'
                    .format(epoch=epoch, log_likelihood_mental=log_likelihood_mental))

        scheduler.step()
        
    """ After training, use test dataset to check the accuracy of the predicted hazard rate. """
    for batch in tqdm(testing_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
            mental_time, mental_type, action_time, action_type = map(lambda x: x.to(opt.device), batch)
            # print('----- mental_time ----')
            # print(mental_time)
            
            enc_out, haz_out = model(action_time, action_type)
            haz_out = haz_out.squeeze(-1)
            # print('----- haz_out ----')
            # print(haz_out)

            mental_time_list = mental_time.tolist()
            haz_out_list = haz_out.tolist()
            
            
            
            for seq_idx in range(len(haz_out_list)):
                
                # print('----- mental_time_list[seq_idx] ----')
                # print(mental_time_list[seq_idx])
                # print('----- haz_out_list[seq_idx] ----')
                # print(haz_out_list[seq_idx])
                
                plt.plot(range(len(haz_out_list[seq_idx])), haz_out_list[seq_idx], color='blue', label='learned hazard rate')
                                
                nonzero_indices = [index for index, value in enumerate(mental_time_list[seq_idx]) if value != 0]
                selected_values = [haz_out_list[seq_idx][i] for i in nonzero_indices]
                
                plt.scatter(nonzero_indices, selected_values, marker='o', color='red')
                plt.xlabel('time')
                plt.ylabel('hazard function')
                plt.title('predicted hazard function of mental event')
                plt.savefig('learned_hazard_rate_{}seq_10epoch.png'.format(seq_idx))
                plt.close()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True) ##### data-hawkes: 8000 seqs, 每个seq中event的个数不等, 5 num_types
    parser.add_argument('-num_types', type=int, default=3) 
    
    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood-mental\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, opt)


if __name__ == '__main__':    
    main()
