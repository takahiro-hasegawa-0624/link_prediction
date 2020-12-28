import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from link_prediction.model.link_predictor import Link_Prediction_Model
from link_prediction.my_util import my_utils

import torch

import optuna

import argparse

def get_optimizer(trial, model, args):
    weight_decay = trial.suggest_loguniform('weight_decay', args.weight_decay_min, args.weight_decay_max)
    lr = trial.suggest_uniform('lr', args.lr_min, args.lr_max)
    
    weight_decay_bias = trial.suggest_loguniform('weight_decay_bias', args.weight_decat_bias_min, args.weight_decat_bias_max)
    lr_bias = trial.suggest_uniform('lr_bias', args.lr_bias_min, args.lr_bias_max)
    
    optimizer = {}
    optimizer['bias'] = torch.optim.Adam(model.model.bias.parameters(), weight_decay=weight_decay_bias, lr=lr_bias)
    optimizer['convs'] = torch.optim.Adam(model.model.convs.parameters(), weight_decay=weight_decay, lr=lr)
    optimizer['lins'] = torch.optim.Adam(model.model.lins.parameters(), weight_decay=weight_decay, lr=lr)
    
    return optimizer

def get_scheduler(trial, model, args):    
    scheduler = {}
    bias_gamma = trial.suggest_uniform('bias_gamma', args.bias_gamma_min, args.bias_gamma_max)
    lins_convs_gamma = trial.suggest_uniform('lins_convs_gamma', args.lins_convs_gamma_min, args.lins_convs_gamma_max)
    scheduler['bias'] = torch.optim.lr_scheduler.ExponentialLR(model.optimizer['bias'], gamma=bias_gamma)

    if args.lins_convs_scheduler == 1:
        scheduler['convs'] = torch.optim.lr_scheduler.StepLR(model.optimizer['convs'], step_size=args.lins_convs_step_size, gamma=lins_convs_gamma)
        scheduler['lins'] = torch.optim.lr_scheduler.StepLR(model.optimizer['lins'], step_size=args.lins_convs_step_size, gamma=lins_convs_gamma)

    return scheduler

def run_objective(args, model):
    def objective(trial):
        model(
            encode_modelname=args.encode_model, 
            decode_modelname=args.decode_model, 
            activation = args.activation, 
            self_loop_mask = True,
            num_hidden_channels = 256, 
            num_layers = 32, 
            hidden_channels = None, 
            dropout = 0.5,
            sigmoid_bias = True,
            negative_sampling_ratio = 1,
            threshold = 0.5
        )
        
        model.my_optimizer(get_optimizer(trial, model, args))
        model.my_scheduler(get_scheduler(trial, model, args))
        
        model.run_training(num_epochs=args.num_epochs, print_log=False, current_dir=os.path.dirname(os.path.abspath(__file__)))
        
        model.model_evaluate(validation=True, save=False)
        
        return 1.0 - model.best_val

    return objective

def main():
    parser = argparse.ArgumentParser(description='execute optuna studying')
    parser.add_argument('--encode_model', type=str, default='GCNII')
    parser.add_argument('--decode_model', type=str, default='GAE')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--timeout', type=int, default=60*60*12)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--weight_decay_min', type=float, default=1e-12)
    parser.add_argument('--weight_decay_max', type=float, default=1e-2)
    parser.add_argument('--weight_decat_bias_min', type=float, default=1e-5)
    parser.add_argument('--weight_decat_bias_max', type=float, default=1e-1)
    parser.add_argument('--lr_min', type=float, default=1e-3)
    parser.add_argument('--lr_max', type=float, default=5e-1)
    parser.add_argument('--lr_bias_min', type=float, default=1e-3)
    parser.add_argument('--lr_bias_max', type=float, default=1e-1)
    parser.add_argument('--bias_gamma_min', type=float, default=0.99)
    parser.add_argument('--bias_gamma_max', type=float, default=0.994)
    parser.add_argument('--lins_convs_gamma_min', type=float, default=0.5)
    parser.add_argument('--lins_convs_gamma_max', type=float, default=1.0)
    parser.add_argument('--lins_convs_step_size', type=int, default=2)
    parser.add_argument('--lins_convs_scheduler', type=int, default=0)

    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    model = Link_Prediction_Model(dataset_name='Cora', val_ratio=0.05, test_ratio=0.1, data_dir=os.path.dirname(current_dir)+'/data')

    study = optuna.create_study()
    study.optimize(run_objective(args, model), timeout=args.timeout)

    model(
        modelname=args.modelname, 
        activation = args.activation, 
        self_loop_mask = True,
        num_hidden_channels = 256, 
        num_layers = 32, 
        hidden_channels = None, 
        dropout = 0.5,
        sigmoid_bias = True,
        negative_sampling_ratio = 1,
        threshold = 0.5
    )

    optimizer = {}
    optimizer['bias'] = torch.optim.Adam(model.decode_model.bias.parameters(), weight_decay=study.best_params['weight_decay_bias'], lr=study.best_params['lr_bias'])
    optimizer['convs'] = torch.optim.Adam(model.encode_model.convs.parameters(), weight_decay=study.best_params['weight_decay'], lr=study.best_params['lr'])
    optimizer['lins'] = torch.optim.Adam(model.encode_model.lins.parameters(), weight_decay=study.best_params['weight_decay'], lr=study.best_params['lr'])
    model.my_optimizer(optimizer)

    scheduler = {}
    scheduler['bias'] = torch.optim.lr_scheduler.ExponentialLR(model.optimizer['bias'], gamma=study.best_params['bias_gamma'])
    scheduler['convs'] = torch.optim.lr_scheduler.MultiStepLR(model.optimizer['convs'], milestones=[1000,2000], gamma=study.best_params['lins_convs_gamma'])
    scheduler['lins'] = torch.optim.lr_scheduler.MultiStepLR(model.optimizer['lins'], milestones=[1000,2000], gamma=study.best_params['lins_convs_gamma'])
    model.my_scheduler(scheduler)
        
    model.run_training(num_epochs=5000, print_log=False, save_dir=current_dir)
    model.model_evaluate(validation=True, save=True)

    return

if __name__ == '__main__':
    main()