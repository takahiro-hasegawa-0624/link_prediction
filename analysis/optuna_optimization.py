import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from link_prediction.model.link_predictor import Link_Prediction_Model
from link_prediction.my_util import my_utils

import torch

import optuna

import argparse

def get_optimizer(trial, model, args):
    optimizer = {}

    weight_decay = trial.suggest_loguniform('weight_decay', args.weight_decay_min, args.weight_decay_max)
    lr = trial.suggest_uniform('lr', args.lr_min, args.lr_max)
    optimizer['encoder_convs'] = torch.optim.Adam(model.encode_model.convs.parameters(), weight_decay=weight_decay, lr=lr)
    optimizer['encoder_lins'] = torch.optim.Adam(model.encode_model.lins.parameters(), weight_decay=weight_decay, lr=lr)

    if args.decode_model != 'Cat_Linear_Decoder':
        weight_decay_bias = trial.suggest_loguniform('weight_decay_bias', args.weight_decay_bias_min, args.weight_decay_bias_max)
        lr_bias = trial.suggest_uniform('lr_bias', args.lr_bias_min, args.lr_bias_max)
        optimizer['decoder_bias'] = torch.optim.Adam(model.decode_model.bias.parameters(), weight_decay=weight_decay_bias, lr=lr_bias)

    if args.decode_model == 'Cat_Linear_Decoder':
        weight_decay_decoder = trial.suggest_loguniform('weight_decay_decoder', args.weight_decay_decoder_min, args.weight_decay_decoder_max)
        lr_decoder = trial.suggest_uniform('lr_decoder', args.lr_decoder_min, args.lr_decoder_max)
        optimizer['decoder_lins'] = torch.optim.Adam(model.decode_model.lins.parameters(), weight_decay=weight_decay_decoder, lr=lr_decoder)
    
    return optimizer

def get_scheduler(trial, model, args):    
    scheduler = {}
    if args.decode_model != 'Cat_Linear_Decoder':
        bias_gamma = trial.suggest_uniform('bias_gamma', args.bias_gamma_min, args.bias_gamma_max)
        scheduler['decoder_bias'] = torch.optim.lr_scheduler.ExponentialLR(model.optimizer['decoder_bias'], gamma=bias_gamma)

    if args.decode_model == 'Cat_Linear_Decoder':
        decoder_gamma = trial.suggest_uniform('decoder_gamma', args.decoder_gamma_min, args.decoder_gamma_max)
        scheduler['decoder_lins'] = torch.optim.lr_scheduler.ExponentialLR(model.optimizer['decoder_lins'], gamma=decoder_gamma)

    if args.lins_convs_scheduler == 1:
        lins_convs_gamma = trial.suggest_uniform('lins_convs_gamma', args.lins_convs_gamma_min, args.lins_convs_gamma_max)
        scheduler['encoder_convs'] = torch.optim.lr_scheduler.StepLR(model.optimizer['encoder_convs'], step_size=args.lins_convs_step_size, gamma=lins_convs_gamma)
        scheduler['encoder_lins'] = torch.optim.lr_scheduler.StepLR(model.optimizer['encoder_lins'], step_size=args.lins_convs_step_size, gamma=lins_convs_gamma)

    return scheduler

def get_gcnii_param(trial, args):
    if args.encode_model == 'GCNII':
        alpha = trial.suggest_uniform('gcnii_alpha', args.gcnii_alpha_min, args.gcnii_alpha_max)
        theta = trial.suggest_uniform('gcnii_theta', args.gcnii_theta_min, args.gcnii_theta_max)
    else:
        alpha = None
        theta = None

    return alpha, theta

def run_objective(args, model):
    def objective(trial):
        sigmoid_bias = (args.decode_model != 'Cat_Linear_Decoder')
        alpha, theta = get_gcnii_param(trial, args)

        model(
            encode_modelname=args.encode_model, 
            decode_modelname=args.decode_model, 
            activation = args.activation, 
            self_loop_mask = True,
            num_hidden_channels = args.num_hidden_channels, 
            num_layers = args.num_layers, 
            hidden_channels = None, 
            dropout = 0.5,
            sigmoid_bias = sigmoid_bias,
            negative_sampling_ratio = 1,
            threshold = 0.5,
            alpha = alpha,
            theta = theta
        )
        
        model.my_optimizer(get_optimizer(trial=trial, model=model, args=args))
        model.my_scheduler(get_scheduler(trial=trial, model=model, args=args))
        
        model.run_training(num_epochs=args.num_epochs, print_log=False, save_dir=os.path.dirname(os.path.abspath(__file__)))
        
        model.model_evaluate(validation=True, save=False)
        
        return 1.0 - model.best_val

    return objective

def main():
    parser = argparse.ArgumentParser(description='execute optuna studying')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--encode_model', type=str, default='GCNII')
    parser.add_argument('--decode_model', type=str, default='Shifted-GAE')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--num_hidden_channels', type=int, default=32)
    parser.add_argument('--timeout', type=int, default=60*60*12)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--weight_decay_min', type=float, default=1e-12)
    parser.add_argument('--weight_decay_max', type=float, default=1e-8)
    parser.add_argument('--weight_decay_bias_min', type=float, default=1e-5)
    parser.add_argument('--weight_decay_bias_max', type=float, default=1e-1)
    parser.add_argument('--weight_decay_decoder_min', type=float, default=1e-10)
    parser.add_argument('--weight_decay_decoder_max', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=1e-3)
    parser.add_argument('--lr_max', type=float, default=5e-1)
    parser.add_argument('--lr_bias_min', type=float, default=1e-3)
    parser.add_argument('--lr_bias_max', type=float, default=1e-1)
    parser.add_argument('--lr_decoder_min', type=float, default=1e-3)
    parser.add_argument('--lr_decoder_max', type=float, default=1e-1)
    parser.add_argument('--bias_gamma_min', type=float, default=0.99)
    parser.add_argument('--bias_gamma_max', type=float, default=0.994)
    parser.add_argument('--decoder_gamma_min', type=float, default=0.99)
    parser.add_argument('--decoder_gamma_max', type=float, default=0.994)
    parser.add_argument('--lins_convs_gamma_min', type=float, default=0.5)
    parser.add_argument('--lins_convs_gamma_max', type=float, default=1.0)
    parser.add_argument('--lins_convs_step_size', type=int, default=2)
    parser.add_argument('--lins_convs_scheduler', type=int, default=0)
    parser.add_argument('--gcnii_alpha_min', type=float, default=0.05)
    parser.add_argument('--gcnii_alpha_max', type=float, default=0.5)
    parser.add_argument('--gcnii_theta_min', type=float, default=0.1)
    parser.add_argument('--gcnii_theta_max', type=float, default=1.5)
    parser.add_argument('--save_study', type=int, default=1)
    parser.add_argument('--read_strage_only', type=int, default=0)

    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    model = Link_Prediction_Model(dataset_name=args.dataset, val_ratio=0.05, test_ratio=0.1, data_dir=os.path.dirname(current_dir)+'/data')

    study_name = f'{args.encode_model}_{args.decode_model}_num_layers_{args.num_layers}_num_hidden_channels_{args.num_hidden_channels}_num_epochs_{args.num_epochs}'

    if args.save_study==1:
        study = optuna.create_study(
            study_name=study_name,
            storage='sqlite:///optuna_study.db',
            load_if_exists=True
        )
    else:
        study = optuna.create_study()

    if args.read_strage_only==0:
        study.optimize(run_objective(args=args, model=model), timeout=args.timeout)

    if args.encode_model == 'GCNII':
        alpha = study.best_params['gcnii_alpha']
        theta = study.best_params['gcnii_theta']
    else:
        alpha = None
        theta = None

    model(
        encode_modelname=args.encode_model, 
        decode_modelname=args.decode_model, 
        activation = args.activation, 
        self_loop_mask = True,
        num_hidden_channels = args.num_hidden_channels, 
        num_layers = args.num_layers, 
        hidden_channels = None, 
        dropout = 0.5,
        sigmoid_bias = True,
        negative_sampling_ratio = 1,
        threshold = 0.5,
        alpha=alpha,
        theta=theta
    )

    optimizer = {}
    if args.decode_model != 'Cat_Linear_Decoder':
        optimizer['decoder_bias'] = torch.optim.Adam(model.decode_model.bias.parameters(), weight_decay=study.best_params['weight_decay_bias'], lr=study.best_params['lr_bias'])
    if args.decode_model == 'Cat_Linear_Decoder':
        optimizer['decoder_lins'] = torch.optim.Adam(model.decode_model.lins.parameters(), weight_decay=study.best_params['weight_decay_decoder'], lr=study.best_params['lr_decoder'])
    optimizer['encoder_convs'] = torch.optim.Adam(model.encode_model.convs.parameters(), weight_decay=study.best_params['weight_decay'], lr=study.best_params['lr'])
    optimizer['encoder_lins'] = torch.optim.Adam(model.encode_model.lins.parameters(), weight_decay=study.best_params['weight_decay'], lr=study.best_params['lr'])
    model.my_optimizer(optimizer)

    scheduler = {}
    if args.decode_model != 'Cat_Linear_Decoder':
        scheduler['decoder_bias'] = torch.optim.lr_scheduler.ExponentialLR(model.optimizer['decoder_bias'], gamma=study.best_params['bias_gamma'])
    if args.decode_model == 'Cat_Linear_Decoder':
        scheduler['decoder_lins'] = torch.optim.lr_scheduler.ExponentialLR(model.optimizer['decoder_lins'], gamma=study.best_params['decoder_gamma'])
    if args.lins_convs_scheduler == 1:
        scheduler['encoder_convs'] = torch.optim.lr_scheduler.MultiStepLR(model.optimizer['encoder_convs'], milestones=[1000,2000], gamma=study.best_params['lins_convs_gamma'])
        scheduler['encoder_lins'] = torch.optim.lr_scheduler.MultiStepLR(model.optimizer['encoder_lins'], milestones=[1000,2000], gamma=study.best_params['lins_convs_gamma'])
    model.my_scheduler(scheduler)
        
    model.run_training(num_epochs=5000, print_log=True, save_dir=current_dir)
    model.model_evaluate(validation=True, save=True)

    return

if __name__ == '__main__':
    main()