import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from link_prediction.model.link_predictor import Link_Prediction_Model
from link_prediction.my_util import my_utils

import torch

import argparse

def main():
    parser = argparse.ArgumentParser(description='execute grid search')

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--encode_model', type=str, default='GCNII')
    parser.add_argument('--decode_model', type=str, default='GAE')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_hidden_channels', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--jk_mode', type=str, default='max')
    parser.add_argument('--negative_sampling_ratio', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-10)
    parser.add_argument('--weight_decay_bias', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_bias', type=float, default=5e-2)
    parser.add_argument('--lins_convs_step_size', type=int, default=None)
    parser.add_argument('--lins_convs_gamma', type=float, default=None)
    parser.add_argument('--bias_gamma', type=float, default=0.992)
    parser.add_argument('--print_log', type=int, default=1)

    args = parser.parse_args()

    model = Link_Prediction_Model(dataset_name=args.dataset, val_ratio=0.05, test_ratio=0.1)

    model(
        encode_modelname=args.encode_model, 
        decode_modelname=args.decode_model,
        activation = args.activation, 
        self_loop_mask = True,
        num_hidden_channels = args.num_hidden_channels, 
        num_layers = args.num_layers,
        jk_mode = args.jk_mode,
        hidden_channels = None, 
        negative_injection=False,
        dropout = 0.5,
        sigmoid_bias = True,
        negative_sampling_ratio = args.negative_sampling_ratio,
        threshold = 0.5
    )

    optimizer = {}
    if (args.weight_decay is not None) and (args.lr is not None):
        optimizer['convs'] = torch.optim.Adam(model.encode_model.convs.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        optimizer['lins'] = torch.optim.Adam(model.encode_model.lins.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    if (args.weight_decay_bias is not None) and (args.lr_bias is not None):
        optimizer['bias'] = torch.optim.Adam(model.decode_model.bias.parameters(), weight_decay=args.weight_decay_bias, lr=args.lr_bias)
    model.my_optimizer(optimizer)

    scheduler = {}
    if (args.weight_decay is not None) and (args.lr is not None):
        scheduler['convs'] = torch.optim.lr_scheduler.StepLR(model.optimizer['convs'], step_size=args.lins_convs_step_size, gamma=args.lins_convs_gamma)
        scheduler['lins'] = torch.optim.lr_scheduler.StepLR(model.optimizer['lins'], step_size=args.lins_convs_step_size, gamma=args.lins_convs_gamma)
    if (args.weight_decay_bias is not None) and (args.lr_bias is not None):
        scheduler['bias'] = torch.optim.lr_scheduler.ExponentialLR(model.optimizer['bias'], gamma=args.bias_gamma)
    model.my_scheduler(scheduler)

    if args.print_log == 1:
        print_log = True
    else:
        print_log = False

    model.run_training(num_epochs=args.num_epochs, print_log=print_log, current_dir=os.path.dirname(os.path.abspath(__file__)))
    model.model_evaluate(validation=True, save=True)

    return

if __name__ == '__main__':
    main()