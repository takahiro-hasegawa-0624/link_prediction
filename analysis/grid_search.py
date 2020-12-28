from link_prediction.model.link_predictor import Link_Prediction_Model
from link_prediction.my_util import my_utils

import numpy as np
import torch

import itertools

import argparse

def main():
    parser = argparse.ArgumentParser(description='execute grid search')

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--modelname', type=str, default='GCNII')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--weight_decay_min', type=float, default=1e-12)
    parser.add_argument('--weight_decay_max', type=float, default=1e-2)
    parser.add_argument('--weight_decay_split', type=int, default=11)
    parser.add_argument('--weight_decat_bias_min', type=float, default=1e-5)
    parser.add_argument('--weight_decat_bias_max', type=float, default=1e-1)
    parser.add_argument('--weight_decat_bias_split', type=int, default=5)
    parser.add_argument('--lr_min', type=float, default=1e-3)
    parser.add_argument('--lr_max', type=float, default=5e-1)
    parser.add_argument('--lr_split', type=int, default=10)
    parser.add_argument('--lr_bias_min', type=float, default=1e-3)
    parser.add_argument('--lr_bias_max', type=float, default=1e-1)
    parser.add_argument('--lr_bias_split', type=int, default=6)
    parser.add_argument('--bias_gamma_min', type=float, default=0.99)
    parser.add_argument('--bias_gamma_max', type=float, default=0.994)
    parser.add_argument('--bias_gamma_split', type=int, default=4)

    args = parser.parse_args()

    model = Link_Prediction_Model(dataset_name=args.dataset, val_ratio=0.05, test_ratio=0.1)

    weight_decay_list = np.linspace(np.log10(args.weight_decay_min), np.log10(args.weight_decay_max), args.weight_decay_split)
    weight_decay_list = 10**weight_decay_list
    weight_decat_bias_list = np.linspace(np.log10(args.weight_decay_bias_min), np.log10(args.weight_decay_bias_max), args.weight_decay_bias_split)
    weight_decat_bias_list = 10**weight_decat_bias_list
    lr_list = np.linspace(np.log10(args.lr_min), np.log10(args.lr_max), args.lr_split)
    lr_list = 10**lr_list
    lr_bias_list = np.linspace(np.log10(args.lr_bias_min), np.log10(args.lr_bias_max), args.lr_bias_split)
    lr_bias_list = 10**lr_bias_list
    bias_gamma_list = np.linspace(args.bias_gamma_min, args.bias_gamma_max, args.bias_gamma_split)
    # lins_convs_gamma_list = []

    for weight_decay, weight_decay_bias, lr, lr_bias, bias_gamma in itertools.product(weight_decay_list, weight_decat_bias_list, lr_list, lr_bias_list, bias_gamma_list):
        model(
            modelname=args.modelname, 
            activation = args.activation, 
            self_loop_mask = True,
            num_hidden_channels = 256, 
            num_layers = args.num_layers, 
            hidden_channels = None, 
            dropout = 0.5,
            sigmoid_bias = True,
            negative_sampling_ratio = 1,
            threshold = 0.5
        )
        optimizer = {}
        optimizer['bias'] = torch.optim.Adam(model.decode_model.bias.parameters(), weight_decay=weight_decay_bias, lr=lr_bias)
        optimizer['convs'] = torch.optim.Adam(model.encode_model.convs.parameters(), weight_decay=weight_decay, lr=lr)
        optimizer['lins'] = torch.optim.Adam(model.encode_model.lins.parameters(), weight_decay=weight_decay, lr=lr)
        model.my_optimizer(optimizer)

        scheduler = {}
        scheduler['bias'] = torch.optim.lr_scheduler.ExponentialLR(model.optimizer['bias'], gamma=bias_gamma)
        # scheduler['convs'] = torch.optim.lr_scheduler.MultiStepLR(model.optimizer['convs'], milestones=[1000,2000], gamma=lins_convs_gamma)
        # scheduler['lins'] = torch.optim.lr_scheduler.MultiStepLR(model.optimizer['lins'], milestones=[1000,2000], gamma=lins_convs_gamma)
        model.my_scheduler(scheduler)
            
        model.run_training(num_epochs=4000, print_log=False)
        model.model_evaluate(validation=True, save=True)

        return

if __name__ == '__main__':
    main()