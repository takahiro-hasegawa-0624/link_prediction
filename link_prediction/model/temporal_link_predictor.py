"""
graph neural networks for link prediction

    link preiction モデルのclassを定義する

Todo:

"""

import os, sys
import shutil
import datetime
import cloudpickle

import random
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F

from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling

from link_prediction.model.temporal_node_encoder import GCRN, GCRNII, EvolveGCNO, EvolveGCNIIO
from link_prediction.model.link_decoder import GAE, VGAE, S_VAE, Cat_Linear_Decoder, Mean_Linear_Decoder
from link_prediction.my_util import my_utils

class Temporal_Link_Prediction_Model():
    '''Link_Prediction_Model

    Link_Prediction_Modelの学習と検証を行う

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        data (torch_geometric.data.Data): グラフデータ.
        all_pos_edge_index (torch.Tensor[2, num_pos_edges]): train_test_split前の全リンク.
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
        y_true (numpy.ndarray[num_nodes, num_nodes].flatten()): 全リンクの隣接行列をflattenしたもの.
        y_train (numpy.ndarray[num_nodes, num_nodes].flatten()): trainデータのリンクの隣接行列をflattenしたもの.
        y_train_cpu (numpy.ndarray[num_nodes, num_nodes].flatten()): trainデータのリンクの隣接行列をflattenしたもの. to(device)していない.
        mask (numpy.ndarray[num_nodes, num_nodes].flatten()): validation, testのpos_edge, neg_edgeとしてサンプリングしたリンクをFalse、それ以外をTrueとした隣接行列をflattenしたもの.
        num_neg_edges (int): trainの正例でないノードの組み合わせ総数.

        encode_modelname (:obj:`str`): 'NN', 'GCN', 'GCNII'を選択.
        decode_modelname (:obj:`str`): 'GAE', 'VGAE'を選択.
        model (torch.nn.Module): 上記で定義したNN / GCN / GCNII のいずれか.
        optimizer (dic of torch.optim): optimizerの辞書. tagは'bias' (sigmoidのバイアス項), 'convs' (graph convolution), 'lins' (全結合層)
        activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
        sigmoid_bias (bool): If seto to True, sigmoid関数の入力にバイアス項が加わる (sigmoid(z^tz + b)。 (Default: False)
        self_loop_mask (bool): If seto to True, 特徴量の内積が必ず正となり、必ず存在確率が0.5以上となる自己ループを除外する。 (Default: False)
        num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される.
        num_layers (int or None): 隠れ層の数.
        hidden_channels (list of int, or None): 各隠れ層の出力の配列. 指定するとnum_hidden_channels とnum_layersは無効化される. (Default: None)
        alpha (float): convolution後に初期層を加える割合. (Default: 0.1)
        theta (float): .
        shared_weights (bool): . (Default: True)
        dropout (float): 各層のDropoutの割合. (Default: 0.0)
        negative_sampling_ratio (float or None): 正例に対する負例のサンプリング比率. If set to None, 負例を全て用いる. (Default: None)
        num_negative_samples (int or None): 負例のサンプリング数. 負例を全て用いる場合はNone.
        threshold (float): リンク有りと予測するsigmoidの出力の閾値. (Default: 0.5)

        logs (:obj:`str`): txtファイルとして保存する学習のlog.

        num_epochs (int): epoch数.
        best_model (torch.nn.Module)validationの結果が最も良いモデル.
        best_epoch (int): validationの結果が最も良いepoch数.
        best_val (float): validationスコアの最高値.

        train_loss_list (list of float): trainのlossの配列.
        val_loss_list (list of float): validationのlossの配列.
        test_loss_list (list of float): testのlossの配列.

        train_auc_list (list of float): trainのaucの配列.
        val_auc_list (list of float): validationのaucの配列.
        test_auc_list (list of float): testのaucの配列.

        train_precision_list (list of float): trainのprecisionの配列.
        val_precision_list (list of float): valのprecisionの配列.
        test_precision_list (list of float): testのprecisionの配列.

        summary (pd.DataFrame): 学習結果を集約するDataFrame. csvとして保存.
    '''  
    def __init__(self, dataset_name='Factset_temporal', data=None, val_ratio=0.05, test_ratio=0.1, data_dir='../data'):
        '''
        dataを渡し、train_test_splitを行う.

        Parameters:
            dataset_name (:obj:`str` or None): データセット名.'Cora', 'factset' (Default: '')
            data (torch_geometric.data.Data or None): グラフデータ. (Default: None)
            val_ratio (float): 全体のデータ数に対するvalidationデータの割合. (Default: 0.05)
            test_ratio (float): 全体のデータ数に対するtestデータの割合. (Default: 0.1)
        '''
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.dataset_name = dataset_name

        if val_ratio + test_ratio == 1:
            self.future_prediction = True
            dataset_name='Factset_temporal_future'
        else:
            self.future_prediction = False

        if data is None:
            self.data_list = []
            for year in [2015,2016,2017,2018]:
                self.data_list.append(my_utils.data_downloader(dataset = f'Factset_{year}', data_dir=data_dir, data_type='dynamic'))

        self.all_pos_edge_index, self.train_pos_edge_adj_t, self.y_true, self.y_train_cpu, self.mask = my_utils.data_processor(data=self.data_list[-1], undirected=True, val_ratio=val_ratio, test_ratio=test_ratio)

        for i in range(len(self.data_list)):
            self.data_list[i] = self.data_list[i].to(self.device)

        self.all_pos_edge_index = self.all_pos_edge_index.to(self.device)
        self.train_pos_edge_adj_t = self.train_pos_edge_adj_t.to(self.device)
        self.y_true = self.y_true.to(self.device)
        self.y_train = self.y_train_cpu.to(self.device)
        self.mask = self.mask.to(self.device)
        self.edge_index_for_negative_sampling = torch.cat([self.data_list[-1].train_pos_edge_index, self.data_list[-1].test_neg_edge_index, self.data_list[-1].test_pos_edge_index, self.data_list[-1].val_neg_edge_index, self.data_list[-1].val_pos_edge_index], dim = -1)
        self.num_neg_edges = self.data_list[-1].num_nodes*(self.data_list[-1].num_nodes-1)/2 - self.data_list[-1].train_pos_edge_index.size(1)
        
        # self.shuffled_negative_edge_index = torch.cat([torch.Tensor(np.array(list(itertools.combinations(range(self.data_list[-1].num_nodes), 2))).T).to(self.device), self.edge_index_for_negative_sampling], dim=-1)
        # self.shuffled_negative_edge_index, counts = torch.unique(self.shuffled_negative_edge_index, return_counts=True, dim=-1)
        # self.shuffled_negative_edge_index = self.shuffled_negative_edge_index[:,counts==1]
        # self.shuffled_negative_edge_index = self.shuffled_negative_edge_index[:,np.random.permutation(self.shuffled_negative_edge_index.size(1))]
        # self.start_shuffled_negative_edge_index = 0

        self.data_x_seq = []
        self.edge_index_seq = []
        for i in range(len(self.data_list)):
            self.data_x_seq.append(self.data_list[i].x)

            if i < len(self.data_list)-1:
                (row, col), N, E = self.data_list[i].edge_index, self.data_list[-1].num_nodes, self.data_list[i].edge_index.size(1)
                perm = (col * N + row).argsort()
                row, col = row[perm], col[perm]

                value = None
                for key in ['edge_weight', 'edge_attr', 'edge_type']:
                    if self.data_list[i][key] is not None:
                        value = self.data_list[i][key][perm]
                        break

                for key, item in self.data_list[i]:
                    if item.size(0) == E:
                        self.data_list[i][key] = item[perm]

                edge_adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N), is_sorted=True)
                self.edge_index_seq.append(edge_adj_t)
            else:
                self.edge_index_seq.append(self.train_pos_edge_adj_t)
        print(f"data has been sent to {self.device}.")

    def __call__(self, 
                encode_modelname,
                decode_modelname,
                activation = None, 
                sigmoid_bias = False,
                sigmoid_bias_initial_value=0,
                self_loop_mask = False, 
                num_hidden_channels = None, 
                num_layers = None, 
                hidden_channels = None, 
                negative_injection = False,
                jk_mode = 'cat', 
                alpha = 0.1, 
                theta = 0.5, 
                shared_weights = True, 
                dropout = 0.0,
                negative_sampling_ratio = 1,
                threshold = 0.5,
                seed=42):
        '''
        modelを指定する.

        Parameters:
            encode_modelname (:obj:`str`): 'NN', 'GCN', 'GCNII'を選択.
            decode_modelname (:obj:`str`): 'GAE', 'VGAE'を選択.
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            sigmoid_bias (bool): If set to to True, sigmoid関数の入力にバイ**/uアス項が加わる (sigmoid(z^tz + b)。 (Default: False)
            self_loop_mask (bool): If set to to True, 特徴量の内積が必ず正となり、必ず存在確率が0.5以上となる自己ループを除外する。 (Default: False)
            num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される.
            num_layers (int or None): 隠れ層の数.
            hidden_channels (list of int, or None): 各隠れ層の出力の配列. 指定するとnum_hidden_channels とnum_layersは無効化される. (Default: None)
            negative_injection (bool): If set to to True, negative samplingされたedgeをconvolutionに含める.
            jk_mode (:obj:`str`): JK-Netにおけるaggregation方法. ('cat', 'max' or 'lstm'). (Default: 'cat)
            alpha (float): convolution後に初期層を加える割合. (Default: 0.1)
            theta (float): .
            shared_weights (bool): . (Default: True)
            dropout (float): 各層のDropoutの割合. (Default: 0.0)
            negative_sampling_ratio (float or None): 正例に対する負例のサンプリング比率. If set to None, 負例を全て用いる. (Default: 1)
            threshold (float): リンク有りと予測するsigmoidの出力の閾値. (Default: 0.5)
            seed (int): random seed. (Default: 42)
        '''

        print('######################################')

        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        print(f'reset the model and the random seed {seed}.')

        self.encode_modelname = encode_modelname
        self.decode_modelname = decode_modelname
        self.encode_model = None
        self.self_loop_mask = self_loop_mask            

                            
        if self.encode_modelname == 'GCRN':
            self.encode_model = GCRN(
                data_list = self.data_list,
                decode_modelname = self.decode_modelname,
                train_pos_edge_adj_t = self.train_pos_edge_adj_t,
                num_hidden_channels = num_hidden_channels, 
                num_layers = num_layers, 
                hidden_channels = hidden_channels,
                activation = activation,
                dropout = dropout,
                future_prediction = self.future_prediction
            ).to(self.device)

        if self.encode_modelname == 'GCRNII':
            self.encode_model = GCRNII(
                data_list = self.data_list,
                decode_modelname = self.decode_modelname,
                train_pos_edge_adj_t = self.train_pos_edge_adj_t,
                num_hidden_channels = num_hidden_channels, 
                num_layers = num_layers, 
                alpha=alpha, 
                theta=theta, 
                shared_weights=True, 
                activation = activation,
                dropout = dropout,
                future_prediction = self.future_prediction
            ).to(self.device)
        
        elif self.encode_modelname == 'EvolveGCNO':
            self.encode_model = EvolveGCNO(
                data_list = self.data_list,
                decode_modelname = self.decode_modelname,
                train_pos_edge_adj_t = self.train_pos_edge_adj_t,
                num_hidden_channels = num_hidden_channels, 
                num_layers = num_layers, 
                hidden_channels = hidden_channels,
                activation = activation,
                dropout = dropout,
                future_prediction = self.future_prediction
            ).to(self.device)

        elif self.encode_modelname == 'EvolveGCNIIO':
            self.encode_model = EvolveGCNIIO(
                data_list = self.data_list,
                decode_modelname = self.decode_modelname,
                train_pos_edge_adj_t = self.train_pos_edge_adj_t,
                num_hidden_channels=num_hidden_channels, 
                num_layers=num_layers, 
                alpha=alpha, 
                theta=theta, 
                shared_weights=True, 
                activation = activation,
                dropout=dropout,
                future_prediction = self.future_prediction
            ).to(self.device)

        if self.decode_modelname == 'GAE':
            self.decode_model = GAE(
                encoder = self.encode_model,
                self_loop_mask = self_loop_mask,
                sigmoid_bias = False,
                sigmoid_bias_initial_value=sigmoid_bias_initial_value
            ).to(self.device)

        elif self.decode_modelname == 'Shifted-GAE':
            self.decode_model = GAE(
                encoder = self.encode_model,
                self_loop_mask = self_loop_mask,
                sigmoid_bias = True,
                sigmoid_bias_initial_value=sigmoid_bias_initial_value
            ).to(self.device)

        encoder_module_list=[]
        for name, _ in self.encode_model.named_modules():
            encoder_module_list.append(name)
        decoder_module_list=[]
        for name, _ in self.decode_model.named_modules():
            decoder_module_list.append(name)

        # optimizerはAdamをdefaultとする。self.my_optimizerで指定可能。    
        self.optimizer = {}
        if ('bias' in decoder_module_list) and (len(self.decode_model.bias)>0):
            self.optimizer['decoder_bias'] = torch.optim.Adam(self.decode_model.bias.parameters(), weight_decay=0.01, lr=0.05)
        if ('lins' in decoder_module_list) and (len(self.decode_model.lins)>0):
            self.optimizer['decoder_lins'] = torch.optim.Adam(self.decode_model.lins.parameters(), weight_decay=0.01, lr=0.005)
        if ('convs' in encoder_module_list) and (len(self.encode_model.convs)>0):
            self.optimizer['encoder_convs'] = torch.optim.Adam(self.encode_model.convs.parameters(), weight_decay=0.01, lr=0.005)
        if ('lins' in encoder_module_list) and (len(self.encode_model.lins)>0):
            self.optimizer['encoder_lins'] = torch.optim.Adam(self.encode_model.lins.parameters(), weight_decay=0.01, lr=0.005)

        # learning rate のscheduler はself.my_schedulerで指定可能。
        self.scheduler = {}
        self.scheduler['decoder_bias'] = None
        self.scheduler['decoder_lins'] = None
        self.scheduler['encoder_convs'] = None
        self.scheduler['encoder_lins'] = None

        self.num_hidden_channels = num_hidden_channels
        self.num_layers = self.encode_model.num_layers
        self.hidden_channels = hidden_channels
        self.negative_injection = negative_injection
        self.alpha = alpha
        self.theta = theta
        self.shared_weights = shared_weights
        self.activation = activation
        self.dropout = dropout
        self.sigmoid_bias = sigmoid_bias
        self.threshold = threshold
        self.negative_sampling_ratio = negative_sampling_ratio

        if negative_sampling_ratio is None:
            self.num_negative_samples = None
        else:
            self.num_negative_samples = int(self.negative_sampling_ratio * self.data_list[-1].train_pos_edge_index.size(1))

        self.num_epochs = 0
        self.best_epoch = 0
        self.best_val = 0.0

        # self.feature_distances_list = []

        self.sigmoid_bias_list = []

        self.train_loss_list = []
        self.val_loss_list = []
        self.test_loss_list = []

        self.train_auc_list = []
        self.val_auc_list = []
        self.test_auc_list = []

        self.train_precision_list = []
        self.val_precision_list = []
        self.test_precision_list = []
        
        self.logs=''

        print(f'encode_model: {self.encode_modelname}')
        print(f'decode_model: {self.decode_modelname}')
        print(f'num_layers: {self.num_layers}')
        print(f'activation: {self.activation}')
        print(f'sigmoid_bias: {self.sigmoid_bias}')
        print(f'negative_injection: {self.negative_injection}')
        print(f'negative_sampling_ratio: {self.negative_sampling_ratio}')
        print('ready to train!\n')

    def my_optimizer(self, optimizer):
        '''
        optimizerを指定する.
        trainを開始した後では、変更しない.

        Parameters:
            optimizer (dic of torch.optim): optimizerの辞書. tagは'bias' (sigmoidのバイアス項), 'convs' (graph convolution), 'lins' (全結合層).

        '''
        if self.num_epochs != 0:
            print('unable to change the optimizer while training.')
        else:
            for key, value in optimizer.items():
                self.optimizer[key] = value

    def my_scheduler(self, scheduler):
        '''
        schedulerを指定する.
        trainを開始した後では、変更しない.

        Parameters:
            scheduler (dict of torch.optim.lr_scheduler): scheduler. tagは'bias' (sigmoidのバイアス項), 'convs' (graph convolution), 'lins' (全結合層).
        '''
        # if self.num_epochs != 0:
        #     print('unable to set a scheduler while training.')
        # else:
        if True:
            for key, value in scheduler.items():
                self.scheduler[key] = value

    def train(self):
        '''
        link predictionのmodelを学習する

        Returns:
            loss (float): binary cross entropy
            link_labels (numpy.ndarray[num_train_node + num_negative_samples]): trainデータのedge_indexに対応する正解ラベル. to(device)していない.
            link_probs (torch.Tensor[num_train_node + num_negative_samples]): trainデータのedge_indexに対応する存在確率. to(device)していない.
            z (torch.Tensor[num_train_node, output_channels]): ノードの特徴量テンソル. to(device)していない.
        '''
        self.encode_model.train()
        self.decode_model.train()
    
        for optimizer in self.optimizer.values():
            optimizer.zero_grad()

        edge_index_seq = []
        neg_edge_index_seq = []
        for i in range(len(self.data_list)-1):
            neg_edge_index = negative_sampling(
                edge_index = self.data_list[i].edge_index,
                num_nodes = self.data_list[i].num_nodes,
                num_neg_samples = int(self.data_list[i].edge_index.size(1)*self.negative_sampling_ratio)
            )

            edge_index_seq.append(torch.cat([self.data_list[i].edge_index, neg_edge_index], dim = -1))
            neg_edge_index_seq.append(neg_edge_index)

        neg_edge_index = negative_sampling(
            edge_index = self.edge_index_for_negative_sampling,
            num_nodes = self.data_list[-1].num_nodes,
            num_neg_samples = self.num_negative_samples
        )

        edge_index_seq.append(torch.cat([self.data_list[-1].train_pos_edge_index, neg_edge_index], dim = -1))
        neg_edge_index_seq.append(neg_edge_index)

        z_seq = self.decode_model.encode(self.data_x_seq, self.edge_index_seq)

        link_probs = self.decode_model.decode(z_seq[-1], decode_node_pairs = edge_index_seq[-1])
        if torch.isnan(link_probs).sum()>0:
            print('np.nan occurred')
            link_probs[torch.isnan(link_probs)]=0.5

        link_labels = my_utils.get_link_labels(self.data_list[-1].train_pos_edge_index, neg_edge_index_seq[-1]).to(self.device)
        weight = my_utils.get_loss_weight(self.data_list[-1].train_pos_edge_index, neg_edge_index_seq[-1], self.negative_sampling_ratio).to(self.device)

        for t, z in enumerate(z_seq):
            if t == len(z_seq)-2:
                break

            link_probs_ = self.decode_model.decode(z, decode_node_pairs = edge_index_seq[t])
            link_probs = torch.cat([link_probs, link_probs_], dim=-1)
            if torch.isnan(link_probs).sum()>0:
                print('np.nan occurred')
                link_probs[torch.isnan(link_probs)]=0.5

            link_labels_ = my_utils.get_link_labels(self.data_list[t].edge_index, neg_edge_index_seq[t]).to(self.device)
            link_labels = torch.cat([link_labels, link_labels_], dim=-1)
            weight_ = my_utils.get_loss_weight(self.data_list[t].edge_index, neg_edge_index_seq[t], self.negative_sampling_ratio).to(self.device)
            weight = torch.cat([weight, weight_], dim=-1)

        loss = F.binary_cross_entropy(link_probs, link_labels, weight = weight)
        loss.backward()
        for optimizer in self.optimizer.values():
            if optimizer is not None:
                try:
                    optimizer.step()
                except:
                    pass

        for scheduler in self.scheduler.values():
            if scheduler is not None:
                try:
                    scheduler.step()
                except:
                    pass

        return float(loss.cpu()), link_labels.cpu(), link_probs.cpu().detach().clone(), z.cpu().detach().clone()
            
    @torch.no_grad()
    def val(self):
        '''
        link predictionのmodelをvalidationする.
        torch_geometric.data.Data のval_pos_edge_index とval_neg_edge_index を用いる.

        Returns:
            loss (float): binary cross entropy.
            link_labels (numpy.ndarray[num_validation_pos_node + num_validation_neg_node]): validationデータのリedge_indexに対応する正解ラベル. to(device)していない.
            link_probs (torch.Tensor[num_validation_pos_node + num_validation_neg_node]): validationデータのリedge_indexに対応する正解ラベル. to(device)していない.
        '''
        self.encode_model.eval()
        self.decode_model.eval()
        
        pos_edge_index = self.data_list[-1]['val_pos_edge_index']
        neg_edge_index = self.data_list[-1]['val_neg_edge_index']
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
        
        if pos_edge_index.size(1)==0:
            return None, None, None

        z = self.decode_model.encode(self.data_x_seq, self.edge_index_seq)[-1]
        link_probs = self.decode_model.decode(z, decode_node_pairs = edge_index)
        
        if torch.isnan(link_probs).sum()>0:
            print('np.nan occurred')
            link_probs[torch.isnan(link_probs)]=0.5
        link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).to(self.device)

        loss = F.binary_cross_entropy(link_probs, link_labels)
        
        return float(loss.cpu()), link_labels.cpu(), link_probs.cpu()

    @torch.no_grad()
    def test(self):
        '''
        link predictionのmodelをtestする.
        torch_geometric.data.Data のtest_pos_edge_index とtestneg_edge_index を用いる.

        Returns:
            loss (float): binary cross entropy.
            link_labels (numpy.ndarray[num_test_pos_node + num_test_neg_node]): testデータのリedge_indexに対応する正解ラベル. to(device)していない.
            link_probs (torch.Tensor[num_test_pos_node + num_test_neg_node]): testデータのedge_indexに対応する予測確率. to(device)していない.
        '''
        self.encode_model.eval()
        self.decode_model.eval()
        
        pos_edge_index = self.data_list[-1]['test_pos_edge_index']
        neg_edge_index = self.data_list[-1]['test_neg_edge_index']
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)

        z = self.decode_model.encode(self.data_x_seq, self.edge_index_seq)[-1]
        link_probs = self.decode_model.decode(z, decode_node_pairs = edge_index)

        if torch.isnan(link_probs).sum()>0:
            print('np.nan occurred')
            link_probs[torch.isnan(link_probs)]=0.5
        link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).to(self.device)

        loss = F.binary_cross_entropy(link_probs, link_labels)
        
        return float(loss.cpu()), link_labels.cpu(), link_probs.cpu()

    def run_training(self, num_epochs, print_log = True, save_dir='.'):
        '''
        指定したepoch数、modelを学習する。

        Parameters:
            num_epochs (int): ephch数. 
            print_log (bool): If set to True, 各epochのlossとscoreを表示する. (Default: True)
        '''
        start_epoch = self.num_epochs
        self.num_epochs += num_epochs

        self.start_time = datetime.datetime.now()

        self.base_dir = save_dir
        self.save_dir = f"{self.base_dir}/output/{self.dataset_name}/{self.encode_modelname}/{self.decode_modelname}/activation_{self.activation}/sigmoidbias_{'True' if self.sigmoid_bias is True else 'False'}/numlayers_{self.num_layers}/negative_sampling_ratio_{self.negative_sampling_ratio}/epochs_{self.num_epochs}/{self.start_time.strftime('%Y%m%d_%H%M')}"

        self.path_best_model = self.save_dir + f"/usebestmodel_True"
        if not os.path.isdir(self.path_best_model):
            os.makedirs(self.path_best_model)

        self.path_last_model = self.save_dir + f"/usebestmodel_False"
        if not os.path.isdir(self.path_last_model):
            os.makedirs(self.path_last_model)

        # self.incorrect_edge_index = torch.zeros(2,0).to(self.device)

        for epoch in range(start_epoch+1, self.num_epochs+1):
            # if (self.shuffled_negative_edge_index.size(1)-self.start_shuffled_negative_edge_index)<=self.num_negative_samples:
            #     shuffled_negative_edge_index = torch.cat([torch.Tensor(np.array(list(itertools.combinations(range(self.data.num_nodes), 2))).T).to(self.device), self.edge_index_for_negative_sampling], dim=-1)
            #     shuffled_negative_edge_index, counts = torch.unique(shuffled_negative_edge_index, return_counts=True, dim=-1)
            #     shuffled_negative_edge_index = shuffled_negative_edge_index[:,counts==1]
            #     shuffled_negative_edge_index = shuffled_negative_edge_index[:,np.random.permutation(shuffled_negative_edge_index.size(1))]

            #     self.shuffled_negative_edge_index = torch.cat([self.shuffled_negative_edge_index[:,self.start_shuffled_negative_edge_index:], shuffled_negative_edge_index[:,np.random.permutation(shuffled_negative_edge_index.size(1))]], dim=-1)
            #     self.start_shuffled_negative_edge_index = 0

            train_loss, train_link_labels, train_link_probs, _ = self.train()
            val_loss, val_link_labels, val_link_probs = self.val()
            test_loss, test_link_labels, test_link_probs = self.test()

            # pred_flag = train_link_probs>self.threshold
            # self.incorrect_edge_index = self.train_edge_index[:,train_link_labels != pred_flag]

            if self.sigmoid_bias is True:
                bias = float(self.decode_model.state_dict()['bias.0.bias'].cpu().detach().clone())
                self.sigmoid_bias_list.append(bias)

            train_auc = roc_auc_score(train_link_labels, train_link_probs)
            val_auc = roc_auc_score(val_link_labels, val_link_probs)
            test_auc = roc_auc_score(test_link_labels, test_link_probs)

            train_precision = precision_score(train_link_labels, (train_link_probs>self.threshold), zero_division=0)
            val_precision = precision_score(val_link_labels, (val_link_probs>self.threshold), zero_division=0)
            test_precision = precision_score(test_link_labels, (test_link_probs>self.threshold), zero_division=0)

            # tmp_index = np.arange(z.shape[0])
            # xx, yy = np.meshgrid(tmp_index, tmp_index)
            # distances = np.linalg.norm(z[xx]-z[yy], axis=2)
            # self.feature_distances_list.append(distances.mean())
            
            self.train_loss_list.append(train_loss)
            self.val_loss_list.append(val_loss)
            self.test_loss_list.append(test_loss)

            self.train_auc_list.append(train_auc)
            self.val_auc_list.append(val_auc)
            self.test_auc_list.append(test_auc)

            self.train_precision_list.append(train_precision)
            self.val_precision_list.append(val_precision)
            self.test_precision_list.append(test_precision)
            
            # validationデータによる評価が良いモデルを保存
            if (val_auc > self.best_val) and (val_precision!=0):
                self.best_val = val_auc
                self.best_epoch = epoch

                if print_log is True:
                    print(f'Best Epoch: {self.best_epoch}, Val AUC: {val_auc}, Val Precision: {val_precision}, Test AUC: {test_auc}, Test Precision: {test_precision}')
                with open(f"{self.path_best_model}/best_model.pkl", 'wb') as f:
                    cloudpickle.dump(self.decode_model, f)

            log = 'Epoch: {:03d}/{:03d}, Train_loss: {:.4f}, Val_loss: {:.4f}, Val_Score: {:.4f}, Best_Val_Score: {:.4f}, (Test_loss: {:.4f}, Test_score: {:.4f})\n'
            log = log.format(epoch, self.num_epochs, train_loss, val_loss, val_auc, self.best_val, test_loss, test_auc)
            self.logs += log
            if print_log == 2:
                print(log, end='')

        # with open(f"{self.path_last_model}/model.pkl", 'wb') as f:
        #     cloudpickle.dump(self.decode_model, f)
        
        with open(f"{self.path_best_model}/best_model.pkl", 'rb') as f:
            self.best_decode_model = cloudpickle.load(f)

        print('train completed.\n')

    def read_best_model(self, path):
        with open(path, 'rb') as f:
            self.best_decode_model = cloudpickle.load(f)

    @torch.no_grad()
    def model_evaluate(self, validation=False, save=True, fig_show=True, fig_size=4):
        '''
        学習済みモデルを評価する
        ROC曲線・AUC score・特徴量のcos類似度・特徴量のノルム・混同行列を計算する

        Parameters:
            validation (bool): If set to True, validation scoreが最大のモデルにより評価を行う。 (Default: False)
            save (bool): If set to True, テキストとグラフを保存する。 (Default True)
        '''

        if validation is True:
            self.best_decode_model.eval()
            z = self.best_decode_model.encode(self.data_x_seq, self.edge_index_seq)[-1]
            epochs = self.best_epoch
            path = self.path_best_model

        else:
            self.encode_model.eval()
            self.decode_model.eval()
            z = self.decode_model.encode(self.data_x_seq, self.edge_index_seq)[-1]
            epochs = self.num_epochs
            path = self.path_last_model
        
        z = z.cpu().detach().clone().numpy()
        z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
        z_normalized = z/z_norm
        inner_product = np.dot(z_normalized, z_normalized.T)

        if validation is True:
            pos_edge_index = self.data_list[-1]['val_pos_edge_index']
            neg_edge_index = self.data_list[-1]['val_neg_edge_index']
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
            val_link_probs = self.best_decode_model.decode(self.best_decode_model.encode(self.data_x_seq, self.edge_index_seq)[-1], decode_node_pairs=edge_index).cpu().detach().clone()
            val_link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).cpu()

            pos_edge_index = self.data_list[-1]['test_pos_edge_index']
            neg_edge_index = self.data_list[-1]['test_neg_edge_index']
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
            test_link_probs = self.best_decode_model.decode(self.best_decode_model.encode(self.data_x_seq, self.edge_index_seq)[-1], decode_node_pairs=edge_index).cpu().detach().clone()
            test_link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).cpu()

        else:
            pos_edge_index = self.data_list[-1]['val_pos_edge_index']
            neg_edge_index = self.data_list[-1]['val_neg_edge_index']
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
            val_link_probs = self.decode_model.decode(self.decode_model.encode(self.data_x_seq, self.edge_index_seq)[-1], decode_node_pairs=edge_index).cpu().detach().clone()
            val_link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).cpu()

            pos_edge_index = self.data_list[-1]['test_pos_edge_index']
            neg_edge_index = self.data_list[-1]['test_neg_edge_index']
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
            test_link_probs = self.decode_model.decode(self.decode_model.encode(self.data_x_seq, self.edge_index_seq)[-1], decode_node_pairs=edge_index).cpu().detach().clone()
            test_link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).cpu()
        
        val_link_labels = val_link_labels[~np.isnan(val_link_probs)]
        val_link_probs = val_link_probs[~np.isnan(val_link_probs)]
        val_fpr, val_tpr, _ = roc_curve(val_link_labels, val_link_probs)
        val_auc = roc_auc_score(val_link_labels, val_link_probs)

        test_link_labels = test_link_labels[~np.isnan(test_link_probs)]
        test_link_probs = test_link_probs[~np.isnan(test_link_probs)]
        test_fpr, test_tpr, _ = roc_curve(test_link_labels, test_link_probs)
        test_auc = roc_auc_score(test_link_labels, test_link_probs)

        # plot size indicator
        size=fig_size

        parameters = {'axes.titlesize': 10}
        plt.rcParams.update(parameters)
        
        # lossの図示
        fig, ax = plt.subplots(figsize=(size, size*9/16), dpi=150)
        ax.axvline(x=epochs, c='crimson')
        ax.plot(np.arange(1, self.num_epochs+1), self.train_loss_list, label='train')
        ax.plot(np.arange(1, self.num_epochs+1), self.val_loss_list, label='validation')
        ax.plot(np.arange(1, self.num_epochs+1), self.test_loss_list, label='test')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('binary cross entropy')
        ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
        ax.grid()
        if save:
            fig.savefig(path+'/loss.png')
        if ('ipykernel' in sys.modules) and (fig_show is True):
            plt.show()
        else:
            plt.close()

        # AUCの図示
        fig, ax = plt.subplots(figsize=(size, size*9/16), dpi=150)
        ax.axvline(x=epochs, c='crimson')
        ax.plot(np.arange(1, self.num_epochs+1), self.train_auc_list, label='train')
        ax.plot(np.arange(1, self.num_epochs+1), self.val_auc_list, label='validation')
        ax.plot(np.arange(1, self.num_epochs+1), self.test_auc_list, label='test')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('AUC')
        ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
        ax.grid()
        if save:
            fig.savefig(path+'/auc.png')
        if ('ipykernel' in sys.modules) and (fig_show is True):
            plt.show()
        else:
            plt.close()

        # precisionの図示
        fig, ax = plt.subplots(figsize=(size, size*9/16), dpi=150)
        ax.axvline(x=epochs, c='crimson')
        ax.plot(np.arange(1, self.num_epochs+1), self.train_precision_list, label='train')
        ax.plot(np.arange(1, self.num_epochs+1), self.val_precision_list, label='validation')
        ax.plot(np.arange(1, self.num_epochs+1), self.test_precision_list, label='test')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('precision')
        ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
        ax.grid()
        if save:
            fig.savefig(path+'/precision.png')
        if ('ipykernel' in sys.modules) and (fig_show is True):
            plt.show()
        else:
            plt.close()

        # ROC曲線の図示
        fig, ax = plt.subplots(figsize=(size, size), dpi=150)
        ax.plot(val_fpr, val_tpr, label=f'validation AUC={round(val_auc, 3)}')
        ax.plot(test_fpr, test_tpr, label=f'test AUC={round(test_auc, 3)}')
        ax.legend()
        ax.set_xlabel('FPR: False positive rate')
        ax.set_ylabel('TPR: True positive rate')
        ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
        ax.grid()
        if save:
            fig.savefig(path+'/roc.png')
        if ('ipykernel' in sys.modules) and (fig_show is True):
            plt.show()
        else:
            plt.close()

        # sigmoidのバイアス項の推移を図示
        if self.sigmoid_bias is True:
            fig, ax = plt.subplots(figsize=(size, size*9/16), dpi=150)
            ax.axvline(x=epochs, c='crimson')
            ax.plot(np.arange(1, self.num_epochs+1), self.sigmoid_bias_list)
            ax.set_xlabel('epoch')
            ax.set_ylabel('sigmoid bias')
            ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
            ax.grid()
            if save:
                fig.savefig(path+'/sigmoid_bias.png')
        if ('ipykernel' in sys.modules) and (fig_show is True):
            plt.show()
        else:
            plt.close()

        # 特徴量ベクトルのコサイン類似度のヒストグラムを図示
        fig, ax = plt.subplots(figsize=(size, size*9/16), dpi=150)
        inner_product_flatten = inner_product.flatten()
        inner_product_flatten = inner_product_flatten[~np.isinf(inner_product_flatten)]
        ax.hist(inner_product.flatten(), bins=100)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('cosine similarity of the feature vectors')
        ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
        ax.grid(axis='x')
        ax.axes.yaxis.set_visible(False)
        if save:
            fig.savefig(path+'/cos_similarity.png')
        if ('ipykernel' in sys.modules) and (fig_show is True):
            plt.show()
        else:
            plt.close()

        # 特徴量ベクトルのノルムの平均値を図示
        # fig, ax = plt.subplots(figsize=(size, size*9/16), dpi=150)
        # ax.axvline(x=epochs, c='crimson')
        # ax.plot(np.arange(1, self.num_epochs+1), self.feature_distances_list)
        # ax.set_xlabel('epoch')
        # ax.set_ylabel('Average norm')
        # ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
        # ax.grid()
        # if save:
        #     fig.savefig(path+'/average_norm.png')
        # if ('ipykernel' in sys.modules) and (fig_show is True):
        #     plt.show()
        # else:
        #     plt.close()

        # 特徴量ベクトルのノルムのヒストグラムを図示
        fig, ax = plt.subplots(figsize=(size, size*9/16), dpi=150)
        z_norm_flatten = z_norm.flatten()
        z_norm_flatten = z_norm_flatten[~np.isinf(z_norm_flatten)]
        ax.hist(z_norm_flatten, bins=100)
        # ax.set_xscale('log')
        ax.set_xlabel('norms of the feature vectors')
        ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
        ax.grid(axis='x')
        if save:
            fig.savefig(path+'/norm.png')
        if ('ipykernel' in sys.modules) and (fig_show is True):
            plt.show()
        else:
            plt.close()

        # 特徴量ベクトルをt-SNEにより次元削減->実際のリンクとともに図示
        # tsne = TSNE(n_components=2, random_state = 42, perplexity = 30, n_iter = 1000)
        # z_embedded = tsne.fit_transform(z)

        # fig, ax = plt.subplots(figsize=(size, size), dpi=150)
        # for X in z_embedded[self.all_pos_edge_index.cpu().numpy().T]:
        #     ax.plot(X.T[0], X.T[1], c='gray', linewidth=0.05, alpha=0.4)
        # ax.scatter(z_embedded.T[0], z_embedded.T[1], s=1)
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # ax.set_title(f"{self.decode_modelname}, enc: {self.encode_modelname}, layers: {self.num_layers}, hidden: {self.num_hidden_channels}")
        # if save:
        #     fig.savefig(path+'/t-sne.png')
        # if ('ipykernel' in sys.modules) and (fig_show is True):
        #     plt.show()
        # else:
        #     plt.close()

        # 混合行列
        val_c_matrix = confusion_matrix(val_link_labels, (val_link_probs>self.threshold))
        val_c_matrix_str = f'val_c_matrix: TP_{val_c_matrix[1,1]}_FP_{val_c_matrix[0,1]}_TN_{val_c_matrix[0,0]}_FN_{val_c_matrix[1,0]}'

        test_c_matrix = confusion_matrix(test_link_labels, (test_link_probs>self.threshold))
        test_c_matrix_str = f'test_c_matrix: TP_{test_c_matrix[1,1]}_FP_{test_c_matrix[0,1]}_TN_{test_c_matrix[0,0]}_FN_{test_c_matrix[1,0]}'

        # logの出力
        print('#####val#####')
        print(f'AUC: {val_auc}')
        print(val_c_matrix_str)
        print(f'accuracy: {(val_c_matrix[1,1]+val_c_matrix[0,0])/val_c_matrix.sum().sum()}')
        print(f'precision: {val_c_matrix[1,1]/(val_c_matrix[1,1]+val_c_matrix[0,1])}')
        print(f'recall: {val_c_matrix[1,1]/(val_c_matrix[1,1]+val_c_matrix[1,0])}')

        print('#####test#####')
        print(f'AUC: {test_auc}')
        print(test_c_matrix_str)
        print(f'accuracy: {(test_c_matrix[1,1]+test_c_matrix[0,0])/test_c_matrix.sum().sum()}')
        print(f'precision: {test_c_matrix[1,1]/(test_c_matrix[1,1]+test_c_matrix[0,1])}')
        print(f'recall: {test_c_matrix[1,1]/(test_c_matrix[1,1]+test_c_matrix[1,0])}')
        
        attr = 'Attributes:\n'
        for key, value in self.__dict__.items():
            attr += f'{key}:\n{value}\n\n'
        text = f'{self.encode_model}\n\n{self.decode_model}\n\n{attr}{val_c_matrix_str}\n\nval_AUC_{val_auc}\n\n{test_c_matrix_str}\n\ntest_AUC_{test_auc}'

        if save:
            with open(path+'/log.txt', mode='w') as f:
                f.write(text)

        if save is False:
            # delete directories to save outputs
            shutil.rmtree(self.save_dir)

        # 結果集約csvの作成
        path_csv = f'{self.base_dir}/output/summary.csv'
        if os.path.isfile(path_csv):
            self.summary = pd.read_csv(path_csv)

        else:
            self.summary = pd.DataFrame(columns=[
                'datetime', 
                'seed',
                'dataset', 
                'encode_modelname', 
                'decode_modelname',
                'activation', 
                'sigmoid_bias', 
                'negative_injection', 
                'alpha',
                'theta',
                'bias_weight_decay', 
                'bias_lr', 
                'bias_lr_scheduler', 
                'bias_lr_scheduler_gamma', 
                'convs_weight_decay', 
                'convs_lr', 
                'convs_lr_scheduler', 
                'convs_lr_scheduler_gamma',
                'lins_weight_decay', 
                'lins_lr', 
                'lins_lr_scheduler', 
                'lins_lr_scheduler_gamma',
                'recurrents_weight_decay', 
                'recurrents_lr', 
                'num_layers', 
                'hidden_channels', 
                'negative_sampling_ratio', 
                'num_epochs', 
                'validation',
                'best_epoch', 
                'val_auc', 
                'val_true_positive', 
                'val_false_positive', 
                'val_true_negative', 
                'val_false_negative', 
                'val_accuracy', 
                'val_precision', 
                'val_recall', 
                'test_auc', 
                'test_true_positive', 
                'test_false_positive', 
                'test_true_negative', 
                'test_false_negative', 
                'test_accuracy', 
                'test_precision', 
                'test_recall', 
                'path'
            ])

        log_dic = {}
        log_dic['datetime'] = self.start_time
        log_dic['seed'] = self.seed
        log_dic['dataset'] = self.dataset_name
        log_dic['encode_modelname'] = self.encode_modelname
        log_dic['decode_modelname'] = self.decode_modelname
        log_dic['activation'] = self.activation
        log_dic['sigmoid_bias'] = self.sigmoid_bias
        log_dic['negative_injection'] = self.negative_injection
        log_dic['alpha'] = self.alpha
        log_dic['theta'] = self.theta

        if 'decoder_bias' in self.optimizer.keys():
            log_dic['bias_weight_decay'] = self.optimizer['decoder_bias'].param_groups[0]['weight_decay']
            log_dic['bias_lr'] = self.optimizer['decoder_bias'].param_groups[0]['lr']
            if self.scheduler['decoder_bias'] is None:
                log_dic['bias_lr_scheduler'] = None
                log_dic['bias_lr_scheduler_gamma'] = None
            else:
                log_dic['bias_lr_scheduler'] = self.scheduler['decoder_bias'].__class__.__name__
                log_dic['bias_lr_scheduler_gamma'] = self.scheduler['decoder_bias'].gamma
                log_dic['bias_lr'] = self.scheduler['decoder_bias'].base_lrs[0]

        if 'encoder_convs' in self.optimizer.keys():
            log_dic['convs_weight_decay'] = self.optimizer['encoder_convs'].param_groups[0]['weight_decay']
            log_dic['convs_lr'] = self.optimizer['encoder_convs'].param_groups[0]['lr']
            if self.scheduler['encoder_convs'] is None:
                log_dic['convs_lr_scheduler'] = None
                log_dic['convs_lr_scheduler_gamma'] = None
            else:
                log_dic['convs_lr_scheduler'] = self.scheduler['encoder_convs'].__class__.__name__
                log_dic['convs_lr_scheduler_gamma'] = self.scheduler['encoder_convs'].gamma
                log_dic['convs_lr'] = self.scheduler['encoder_convs'].base_lrs[0]

        if 'encoder_lins' in self.optimizer.keys():
            log_dic['lins_weight_decay'] = self.optimizer['encoder_lins'].param_groups[0]['weight_decay']
            log_dic['lins_lr'] = self.optimizer['encoder_lins'].param_groups[0]['lr']
            if self.scheduler['encoder_lins'] is None:
                log_dic['lins_lr_scheduler'] = None
                log_dic['lins_lr_scheduler_gamma'] = None
            else:
                log_dic['lins_lr_scheduler'] = self.scheduler['encoder_lins'].__class__.__name__
                log_dic['lins_lr_scheduler_gamma'] = self.scheduler['encoder_lins'].gamma
                log_dic['lins_lr'] = self.scheduler['encoder_lins'].base_lrs[0]

        if 'encoder_recurrents' in self.optimizer.keys():
            log_dic['recurrents_weight_decay'] = self.optimizer['encoder_recurrents'].param_groups[0]['weight_decay']
            log_dic['recurrents_lr'] = self.optimizer['encoder_recurrents'].param_groups[0]['lr']

        log_dic['num_layers'] = self.num_layers
        log_dic['hidden_channels'] = self.encode_model.hidden_channels_str
        log_dic['negative_sampling_ratio'] = self.negative_sampling_ratio
        log_dic['num_epochs'] = self.num_epochs
        log_dic['validation'] = validation
        if validation is True:
            log_dic['best_epoch'] = self.best_epoch
        else:
            log_dic['best_epoch'] = None
        log_dic['val_auc'] = val_auc
        log_dic['val_true_positive'] = val_c_matrix[1,1]
        log_dic['val_false_positive'] = val_c_matrix[0,1]
        log_dic['val_true_negative'] = val_c_matrix[0,0]
        log_dic['val_false_negative'] = val_c_matrix[1,0]
        log_dic['val_accuracy'] = (val_c_matrix[1,1]+val_c_matrix[0,0])/val_c_matrix.sum().sum()
        log_dic['val_precision'] = val_c_matrix[1,1]/(val_c_matrix[1,1]+val_c_matrix[0,1])
        log_dic['val_recall'] = val_c_matrix[1,1]/(val_c_matrix[1,1]+val_c_matrix[1,0])
        log_dic['test_auc'] = test_auc
        log_dic['test_true_positive'] = test_c_matrix[1,1]
        log_dic['test_false_positive'] = test_c_matrix[0,1]
        log_dic['test_true_negative'] = test_c_matrix[0,0]
        log_dic['test_false_negative'] = test_c_matrix[1,0]
        log_dic['test_accuracy'] = (test_c_matrix[1,1]+test_c_matrix[0,0])/test_c_matrix.sum().sum()
        log_dic['test_precision'] = test_c_matrix[1,1]/(test_c_matrix[1,1]+test_c_matrix[0,1])
        log_dic['test_recall'] = test_c_matrix[1,1]/(test_c_matrix[1,1]+test_c_matrix[1,0])
        log_dic['path'] = f"./output/{self.dataset_name}/{self.encode_modelname}/{self.decode_modelname}/activation_{self.activation}/sigmoidbias_{'True' if self.sigmoid_bias is True else 'False'}/numlayers_{self.num_layers}/negative_sampling_ratio_{self.negative_sampling_ratio}/epochs_{self.num_epochs}/{self.start_time.strftime('%Y%m%d_%H%M')}"
        self.summary = self.summary.append(pd.Series(log_dic), ignore_index=True)
        self.summary.to_csv(path_csv, index=False)

        return None