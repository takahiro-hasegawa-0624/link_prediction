"""
graph neural networks for link prediction

    link preiction モデルのclassを定義する

Todo:

"""

import os
import shutil
import datetime
import cloudpickle

import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GCN2Conv, global_mean_pool, JumpingKnowledge
from torch_geometric.utils import negative_sampling

import networkx as nx

from link_prediction.model.node_encoder import NN, GCN, GCNII, GCNIIwithJK
from link_prediction.model.link_decoder import GAE, VGAE, Cat_Linear_Encoder
from link_prediction.my_util import my_utils

class Link_Prediction_Model():
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

        train_accuracy_list (list of float): trainのaccuracyの配列.
        val_accuracy_list (list of float): valのaccuracyの配列.
        test_accuracy_list (list of float): testのaccuracyの配列.

        summary (pd.DataFrame): 学習結果を集約するDataFrame. csvとして保存.
    '''  
    def __init__(self, dataset_name='', data=None, val_ratio=0.05, test_ratio=0.1, data_dir='../data'):
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

        if data is None:
            self.data = my_utils.data_downloader(dataset = dataset_name, data_dir=data_dir)

        self.all_pos_edge_index, self.train_pos_edge_adj_t, self.y_true, self.y_train_cpu, self.mask = my_utils.data_processor(self.data)

        self.data = self.data.to(self.device)
        self.all_pos_edge_index = self.all_pos_edge_index.to(self.device)
        self.train_pos_edge_adj_t = self.train_pos_edge_adj_t.to(self.device)
        self.y_true = self.y_true.to(self.device)
        self.y_train = self.y_train_cpu.to(self.device)
        self.mask = self.mask.to(self.device)
        self.edge_index_for_negative_sampling = torch.cat([self.data.train_pos_edge_index, self.data.test_neg_edge_index, self.data.test_pos_edge_index, self.data.val_neg_edge_index, self.data.val_pos_edge_index], dim = -1)
        self.num_neg_edges = self.data.num_nodes*(self.data.num_nodes-1)/2 - self.data.train_pos_edge_index.size(1)

        print(f"data has been sent to {self.device}.")

    def __call__(self, 
                encode_modelname,
                decode_modelname,
                activation = None, 
                sigmoid_bias = False,
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
                negative_sampling_ratio = None,
                threshold = 0.5):
        '''
        modelを指定する.

        Parameters:
            encode_modelname (:obj:`str`): 'NN', 'GCN', 'GCNII'を選択.
            decode_modelname (:obj:`str`): 'GAE', 'VGAE'を選択.
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            sigmoid_bias (bool): If set to to True, sigmoid関数の入力にバイアス項が加わる (sigmoid(z^tz + b)。 (Default: False)
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
            negative_sampling_ratio (float or None): 正例に対する負例のサンプリング比率. If set to None, 負例を全て用いる. (Default: None)
            threshold (float): リンク有りと予測するsigmoidの出力の閾値. (Default: 0.5)
        '''

        print('######################################')

        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print('reset the model and the random seed.')

        self.encode_modelname = encode_modelname
        self.decode_modelname = decode_modelname
        self.self_loop_mask = self_loop_mask            

        if self.encode_modelname == 'NN':
            self.encode_model = NN(
                data = self.data,
                decode_modelname = self.decode_modelname,
                num_hidden_channels = num_hidden_channels, 
                num_layers = num_layers, 
                hidden_channels = hidden_channels,
                activation = activation,
                dropout = dropout
            ).to(self.device)
                            
        elif self.encode_modelname == 'GCN':
            self.encode_model = GCN(
                data = self.data,
                decode_modelname = self.decode_modelname,
                train_pos_edge_adj_t = self.train_pos_edge_adj_t,
                num_hidden_channels = num_hidden_channels, 
                num_layers = num_layers, 
                hidden_channels = hidden_channels,
                activation = activation,
                dropout = dropout
            ).to(self.device)

        elif self.encode_modelname == 'GCNII':
            self.encode_model = GCNII(
                data = self.data,
                decode_modelname = self.decode_modelname,
                train_pos_edge_adj_t = self.train_pos_edge_adj_t,
                num_hidden_channels=num_hidden_channels, 
                num_layers=num_layers, 
                alpha=alpha, 
                theta=theta, 
                shared_weights=True, 
                activation = activation,
                dropout=dropout
            ).to(self.device)

        elif self.encode_modelname == 'GCNIIwithJK':
            self.encode_model = GCNIIwithJK(
                data = self.data,
                decode_modelname = self.decode_modelname,
                train_pos_edge_adj_t = self.train_pos_edge_adj_t,
                num_hidden_channels=num_hidden_channels, 
                num_layers=num_layers, 
                jk_mode=jk_mode, 
                alpha=alpha, 
                theta=theta, 
                shared_weights=True, 
                activation = activation,
                dropout=dropout
            ).to(self.device)

        if self.decode_modelname == 'GAE':
            self.decode_model = GAE(
                encoder = self.encode_model,
                self_loop_mask = self_loop_mask,
                sigmoid_bias = sigmoid_bias
            ).to(self.device)

        elif self.decode_modelname == 'VGAE':
            self.decode_model = VGAE(
                encoder = self.encode_model,
                self_loop_mask = self_loop_mask,
                sigmoid_bias = sigmoid_bias
            ).to(self.device)

        elif self.decode_modelname == 'Cat_Linear_Encoder':
            self.decode_model = Cat_Linear_Encoder(
                encoder = self.encode_model, 
                in_channels = num_hidden_channels, 
                hidden_channels = num_hidden_channels//2, 
                out_channels=1, 
                num_layers=2, 
                dropout=0.5, 
                self_loop_mask = True
            ).to(self.device)

        # optimizerはAdamをdefaultとする。self.my_optimizerで指定可能。
        self.optimizer = {}
        if activation == 'tanh':
            self.optimizer['bias'] = torch.optim.Adam(self.decode_model.bias.parameters(), weight_decay=0.2, lr=0.05)
            self.optimizer['convs'] = torch.optim.Adam(self.encode_model.convs.parameters(), weight_decay=0.01, lr=0.005)
            self.optimizer['lins'] = torch.optim.Adam(self.encode_model.lins.parameters(), weight_decay=0.01, lr=0.005)

        else:
            self.optimizer['bias'] = torch.optim.Adam(self.decode_model.bias.parameters(), weight_decay=0.01, lr=0.05)
            self.optimizer['convs'] = torch.optim.Adam(self.encode_model.convs.parameters(), weight_decay=0.01, lr=0.005)
            self.optimizer['lins'] = torch.optim.Adam(self.encode_model.lins.parameters(), weight_decay=0.01, lr=0.005)

        # learning rate のscheduler はself.my_schedulerで指定可能。
        self.scheduler = {}
        self.scheduler['bias'] = None
        self.scheduler['convs'] = None
        self.scheduler['lins'] = None

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
            self.num_negative_samples = self.negative_sampling_ratio * self.data.train_pos_edge_index.size(1)

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

        self.train_accuracy_list = []
        self.val_accuracy_list = []
        self.test_accuracy_list = []
        
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
        if self.num_epochs != 0:
            print('unable to set a scheduler while training.')
        else:
            for key, value in scheduler.items():
                self.scheduler[key] = value

    def train(self):
        '''
        link predictionのmodelを学習する

        Returns:
            loss (float): binary cross entropy
            link_labels (numpy.ndarray[num_train_node + num_negative_samples]): trainデータのリedge_indexに対応する正解ラベル. to(device)していない.
            link_probs (torch.Tensor[num_train_node + num_negative_samples]): trainデータのリedge_indexに対応する正解ラベル. to(device)していない.
            z (torch.Tensor[num_train_node, output_channels]): ノードの特徴量テンソル. to(device)していない.
        '''
        self.encode_model.train()
        self.decode_model.train()
        
        if self.negative_sampling_ratio is None:
            # 全ての負例を計算
            for optimizer in self.optimizer.values():
                optimizer.zero_grad()

            z = self.decode_model.encode(self.data.x)
            link_probs = self.decode_model.decode(z).flatten()
            if torch.isnan(link_probs).sum()>0:
                print('np.nan occurred')
                link_probs[torch.isnan(link_probs)]=1.0

            loss = F.binary_cross_entropy(link_probs, self.y_train, weight = self.mask)
            if self.decode_modelname == 'VGAE':
                loss = loss + (1 / self.data.num_nodes) * self.decode_model.kl_loss()
            loss.backward()
            for optimizer in self.optimizer.values():
                optimizer.step()

            for scheduler in self.scheduler.values():
                scheduler.step()

            return float(loss.cpu()), self.y_train_cpu, link_probs.cpu().detach().clone(), z.cpu().detach().clone()

        else:
            # negative samplingを行う
            for optimizer in self.optimizer.values():
                optimizer.zero_grad()

            neg_edge_index = negative_sampling(
                edge_index = self.edge_index_for_negative_sampling,
                num_nodes = self.data.num_nodes,
                num_neg_samples = self.num_negative_samples)

            edge_index = torch.cat([self.data.train_pos_edge_index, neg_edge_index], dim = -1)

            if self.negative_injection is True:
                (row, col), N, E = edge_index, self.data.num_nodes, edge_index.size(1)
                perm = (col * N + row).argsort()
                row, col = row[perm], col[perm]

                value = None
                for key in ['edge_weight', 'edge_attr', 'edge_type']:
                    if self.data[key] is not None:
                        value = self.data[key][perm]
                        break

                for key, item in self.data:
                    if item.size(0) == E:
                        self.data[key] = item[perm]

                train_pos_edge_adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N), is_sorted=True)
                z = self.decode_model.encode(self.data.x, edge_index=train_pos_edge_adj_t)
            else:
                z = self.decode_model.encode(self.data.x)

            link_probs = self.decode_model.decode(z)[edge_index.cpu().numpy()]
            if torch.isnan(link_probs).sum()>0:
                print('np.nan occurred')
                link_probs[torch.isnan(link_probs)]=1.0

            link_labels = my_utils.get_link_labels(self.data.train_pos_edge_index, neg_edge_index).to(self.device)
            weight = my_utils.get_loss_weight(self.data.train_pos_edge_index, neg_edge_index, self.negative_sampling_ratio).to(self.device)

            loss = F.binary_cross_entropy(link_probs, link_labels, weight = weight)
            if self.decode_modelname == 'VGAE':
                loss = loss + (1 / self.data.num_nodes) * self.decode_model.kl_loss()
            loss.backward()
            for optimizer in self.optimizer.values():
                if optimizer is not None:
                    optimizer.step()

            for scheduler in self.scheduler.values():
                if scheduler is not None:
                    scheduler.step()

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
        
        pos_edge_index = self.data['val_pos_edge_index']
        neg_edge_index = self.data['val_neg_edge_index']
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)
        
        if pos_edge_index.size(1)==0:
            return None, None, None

        link_probs = self.decode_model.encode_decode(self.data.x)[edge_index.cpu().numpy()]
        if torch.isnan(link_probs).sum()>0:
            print('np.nan occurred')
            link_probs[torch.isnan(link_probs)]=1.0
        link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).to(self.device)
        loss = F.binary_cross_entropy(link_probs, link_labels)
        if self.decode_modelname == 'VGAE':
            loss = loss + (1 / self.data.num_nodes) * self.decode_model.kl_loss()
        
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
        
        pos_edge_index = self.data['test_pos_edge_index']
        neg_edge_index = self.data['test_neg_edge_index']
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)

        link_probs = self.decode_model.encode_decode(self.data.x)[edge_index.cpu().numpy()]
        if torch.isnan(link_probs).sum()>0:
            print('np.nan occurred')
            link_probs[torch.isnan(link_probs)]=1.0
        link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).to(self.device)
        loss = F.binary_cross_entropy(link_probs, link_labels)
        if self.decode_modelname == 'VGAE':
            loss = loss + (1 / self.data.num_nodes) * self.decode_model.kl_loss()
        
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

        self.base_path = save_dir + f"/output/{self.dataset_name}/{self.encode_modelname}/{self.decode_modelname}/activation_{self.activation}/sigmoidbias_{'True' if self.sigmoid_bias is True else 'False'}/numlayers_{self.num_layers}/negative_sampling_ratio_{self.negative_sampling_ratio}/epochs_{self.num_epochs}/{self.start_time.strftime('%Y%m%d_%H%M')}"

        self.path_best_model = self.base_path + f"/usebestmodel_True"
        if not os.path.isdir(self.path_best_model):
            os.makedirs(self.path_best_model)

        self.path_last_model = self.base_path + f"/usebestmodel_False"
        if not os.path.isdir(self.path_last_model):
            os.makedirs(self.path_last_model)

        for epoch in range(start_epoch+1, self.num_epochs+1):
            train_loss, train_link_labels, train_link_probs, _ = self.train()
            val_loss, val_link_labels, val_link_probs = self.val()
            test_loss, test_link_labels, test_link_probs = self.test()

            if self.sigmoid_bias is True:
                bias = float(self.decode_model.state_dict()['bias.0.bias'].cpu().detach().clone())
                self.sigmoid_bias_list.append(bias)

            train_auc = roc_auc_score(train_link_labels, train_link_probs)
            val_auc = roc_auc_score(val_link_labels, val_link_probs)
            test_auc = roc_auc_score(test_link_labels, test_link_probs)

            train_accuracy = accuracy_score(train_link_labels, (train_link_probs>self.threshold))
            val_accuracy = accuracy_score(val_link_labels, (val_link_probs>self.threshold))
            test_accuracy = accuracy_score(test_link_labels, (test_link_probs>self.threshold))

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

            self.train_accuracy_list.append(train_accuracy)
            self.val_accuracy_list.append(val_accuracy)
            self.test_accuracy_list.append(test_accuracy)
            
            # validationデータによる評価が良いモデルを保存
            if val_auc > self.best_val:
                self.best_val = val_auc
                self.best_epoch = epoch
                with open(f"{self.path_best_model}/best_model.pkl", 'wb') as f:
                    cloudpickle.dump(self.decode_model, f)

            log = 'Epoch: {:03d}/{:03d}, Train_loss: {:.4f}, Val_loss: {:.4f}, Val_Score: {:.4f}, (Test_loss: {:.4f}, Test_score: {:.4f})\n'
            log = log.format(epoch, self.num_epochs, train_loss, val_loss, val_auc, test_loss, test_auc)
            self.logs += log
            if print_log is True:
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
    def model_evaluate(self, validation=False, save=True):
        '''
        学習済みモデルを評価する
        ROC曲線・AUC score・特徴量のcos類似度・特徴量のノルム・混同行列を計算する

        Parameters:
            validation (bool): If set to True, validation scoreが最大のモデルにより評価を行う。 (Default: False)
            save (bool): If set to True, テキストとグラフを保存する。 (Default True)
        '''

        if validation is True:
            self.best_decode_model.eval()
            z = self.best_decode_model.encode(self.data.x)
            epochs = self.best_epoch
            path = self.path_best_model

        else:
            self.encode_model.eval()
            self.decode_model.eval()
            z = self.decode_model.encode(self.data.x)
            epochs = self.num_epochs
            path = self.path_last_model
        
        z = z.cpu().detach().clone().numpy()
        z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
        z_normalized = z/z_norm
        inner_product = np.dot(z_normalized, z_normalized.T)

        if validation is True:
            # y_pred = self.decode_model.encode_decode(self.data.x).detach().clone().numpy().flatten()

            pos_edge_index = self.data['test_pos_edge_index']
            neg_edge_index = self.data['test_neg_edge_index']
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)

            test_link_probs = self.best_decode_model.encode_decode(self.data.x).cpu().detach().clone()[edge_index.cpu().numpy()].cpu()
            test_link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).cpu()

        else:
            # y_pred = self.best_model.encode_decode(self.data.x).detach().clone().numpy().flatten()
            pos_edge_index = self.data['test_pos_edge_index']
            neg_edge_index = self.data['test_neg_edge_index']
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim = -1)

            test_link_probs = self.decode_model.encode_decode(self.data.x).cpu().detach().clone()[edge_index.cpu().numpy()].cpu()
            test_link_labels = my_utils.get_link_labels(pos_edge_index, neg_edge_index).cpu()

        fpr, tpr, _ = roc_curve(test_link_labels, test_link_probs)
        auc = roc_auc_score(test_link_labels, test_link_probs)

        # lossの図示
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.axvline(x=epochs, c='crimson')
        ax.plot(np.arange(1, self.num_epochs+1), self.train_loss_list, label='train')
        ax.plot(np.arange(1, self.num_epochs+1), self.val_loss_list, label='validation')
        ax.plot(np.arange(1, self.num_epochs+1), self.test_loss_list, label='test')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('binary cross entropy loss')
        ax.set_title(f"Loss ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
        ax.grid()
        if save:
            fig.savefig(path+'/loss.png')
        if 'inline' in plt.get_backend():
            plt.show()

        # AUCの図示
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.axvline(x=epochs, c='crimson')
        ax.plot(np.arange(1, self.num_epochs+1), self.train_auc_list, label='train')
        ax.plot(np.arange(1, self.num_epochs+1), self.val_auc_list, label='validation')
        ax.plot(np.arange(1, self.num_epochs+1), self.test_auc_list, label='test')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('AUC')
        ax.set_title(f"AUC ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
        ax.grid()
        if save:
            fig.savefig(path+'/auc.png')
        if 'inline' in plt.get_backend():
            plt.show()

        # accuracyの図示
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.axvline(x=epochs, c='crimson')
        ax.plot(np.arange(1, self.num_epochs+1), self.train_accuracy_list, label='train')
        ax.plot(np.arange(1, self.num_epochs+1), self.val_accuracy_list, label='validation')
        ax.plot(np.arange(1, self.num_epochs+1), self.test_accuracy_list, label='test')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f"Accuracy ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
        ax.grid()
        if save:
            fig.savefig(path+'/accuracy.png')
        if 'inline' in plt.get_backend():
            plt.show()

        # ROC曲線の図示
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        ax.plot(fpr, tpr)
        ax.set_xlabel('FPR: False positive rate')
        ax.set_ylabel('TPR: True positive rate')
        ax.set_title(f"AUC = {round(auc, 3)} ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
        ax.grid()
        if save:
            fig.savefig(path+'/roc.png')
        if 'inline' in plt.get_backend():
            plt.show()

        # sigmoidのバイアス項の推移を図示
        if self.sigmoid_bias is True:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            ax.axvline(x=epochs, c='crimson')
            ax.plot(np.arange(1, self.num_epochs+1), self.sigmoid_bias_list)
            ax.set_xlabel('epoch')
            ax.set_ylabel('Sigmoid Bias')
            ax.set_title(f"Sigmoid Bias ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
            ax.grid()
            if save:
                fig.savefig(path+'/sigmoid_bias.png')
        if 'inline' in plt.get_backend():
            plt.show()

        # 特徴量ベクトルのコサイン類似度のヒストグラムを図示
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.hist(inner_product.flatten(), bins=100)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('cosine similarity of the feature vectors')
        ax.set_title(f"Cosine Similarity of feature vectors ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
        ax.grid(axis='x')
        if save:
            fig.savefig(path+'/cos_similarity.png')
        if 'inline' in plt.get_backend():
            plt.show()

        # 特徴量ベクトルのノルムの平均値を図示
        # fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        # ax.axvline(x=epochs, c='crimson')
        # ax.plot(np.arange(1, self.num_epochs+1), self.feature_distances_list)
        # ax.set_xlabel('epoch')
        # ax.set_ylabel('Average norm')
        # ax.set_title(f"Average norm of feature vectors ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
        # ax.grid()
        # if save:
        #     fig.savefig(path+'/average_norm.png')
        # if 'inline' in plt.get_backend():
        #     plt.show()

        # 特徴量ベクトルのノルムのヒストグラムを図示
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        z_norm_flatten = z_norm.flatten()
        ax.hist(z_norm_flatten, bins=100)
        ax.set_xscale('log')
        ax.set_xlabel('norms of the feature vectors')
        ax.set_title(f"Norms of feature vectors ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
        ax.grid(axis='x')
        if save:
            fig.savefig(path+'/norm.png')
        if 'inline' in plt.get_backend():
            plt.show()

        # 特徴量ベクトルをt-SNEにより次元削減->実際のリンクとともに図示
        # tsne = TSNE(n_components=2, random_state = 42, perplexity = 30, n_iter = 1000)
        # z_embedded = tsne.fit_transform(z)

        # fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        # for X in z_embedded[self.all_pos_edge_index.cpu().numpy().T]:
        #     ax.plot(X.T[0], X.T[1], c='gray', linewidth=0.05, alpha=0.4)
        # ax.scatter(z_embedded.T[0], z_embedded.T[1], s=1)
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # ax.set_title(f"t-SNE of feature vectors with edge ({self.encode_modelname} / {self.decode_modelname} / activation_{self.activation} / layers_{self.num_layers})")
        # if save:
        #     fig.savefig(path+'/t-sne.png')
        # if 'inline' in plt.get_backend():
        #     plt.show()

        # 混合行列
        c_matrix = confusion_matrix(test_link_labels, (test_link_probs>self.threshold))
        c_matrix_str = f'TP_{c_matrix[1,1]}_FP_{c_matrix[0,1]}_TN_{c_matrix[0,0]}_FN_{c_matrix[1,0]}'

        # logの出力
        print(f'AUC: {auc}')
        print(c_matrix_str)
        print(f'accuracy: {(c_matrix[1,1]+c_matrix[0,0])/c_matrix.sum().sum()}')
        print(f'precision: {c_matrix[1,1]/(c_matrix[1,1]+c_matrix[0,1])}')
        print(f'recall: {c_matrix[1,1]/(c_matrix[1,1]+c_matrix[1,0])}')
        
        attr = 'Attributes:\n'
        for key, value in self.__dict__.items():
            attr += f'{key}:\n{value}\n\n'
        text = f'{self.encode_model}\n\n{self.decode_model}\n\n{attr}{c_matrix_str}\n\nAUC_{auc}'

        if save:
            with open(path+'/log.txt', mode='w') as f:
                f.write(text)

        if save is False:
            # delete directories to save outputs
            shutil.rmtree(self.base_path)

        # 結果集約csvの作成
        path_csv = './output/summary.csv'
        if os.path.isfile(path_csv):
            self.summary = pd.read_csv(path_csv)

        else:
            self.summary = pd.DataFrame(columns=[
                'datetime', 
                'dataset', 
                'encode_modelname', 
                'decode_modelname',
                'activation', 
                'sigmoid_bias', 
                'negative_injection', 
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
                'num_layers', 
                'hidden_channels', 
                'negative_sampling_ratio', 
                'num_epochs', 
                'validation',
                'best_epoch', 
                'auc', 
                'true_positive', 
                'false_positive', 
                'true_negative', 
                'false_negative', 
                'accuracy', 
                'precision', 
                'recall', 
                'path'
            ])

        log_dic = {}
        log_dic['datetime'] = self.start_time
        log_dic['dataset'] = self.dataset_name
        log_dic['encode_modelname'] = self.encode_modelname
        log_dic['decode_modelname'] = self.decode_modelname
        log_dic['activation'] = self.activation
        log_dic['sigmoid_bias'] = self.sigmoid_bias
        log_dic['negative_injection'] = self.negative_injection

        log_dic['bias_weight_decay'] = self.optimizer['bias'].param_groups[0]['weight_decay']
        log_dic['bias_lr'] = self.optimizer['bias'].param_groups[0]['lr']
        if self.scheduler['bias'] is None:
            log_dic['bias_lr_scheduler'] = None
            log_dic['bias_lr_scheduler_gamma'] = None
        else:
            log_dic['bias_lr_scheduler'] = self.scheduler['bias'].__class__.__name__
            log_dic['bias_lr_scheduler_gamma'] = self.scheduler['bias'].gamma
            log_dic['bias_lr'] = self.scheduler['bias'].base_lrs[0]

        log_dic['convs_weight_decay'] = self.optimizer['convs'].param_groups[0]['weight_decay']
        log_dic['convs_lr'] = self.optimizer['convs'].param_groups[0]['lr']
        if self.scheduler['convs'] is None:
            log_dic['convs_lr_scheduler'] = None
            log_dic['convs_lr_scheduler_gamma'] = None
        else:
            log_dic['convs_lr_scheduler'] = self.scheduler['convs'].__class__.__name__
            log_dic['convs_lr_scheduler_gamma'] = self.scheduler['convs'].gamma
            log_dic['convs_lr'] = self.scheduler['convs'].base_lrs[0]

        log_dic['lins_weight_decay'] = self.optimizer['lins'].param_groups[0]['weight_decay']
        log_dic['lins_lr'] = self.optimizer['lins'].param_groups[0]['lr']
        if self.scheduler['lins'] is None:
            log_dic['lins_lr_scheduler'] = None
            log_dic['lins_lr_scheduler_gamma'] = None
        else:
            log_dic['lins_lr_scheduler'] = self.scheduler['lins'].__class__.__name__
            log_dic['lins_lr_scheduler_gamma'] = self.scheduler['lins'].gamma
            log_dic['lins_lr'] = self.scheduler['lins'].base_lrs[0]

        log_dic['num_layers'] = self.num_layers
        log_dic['hidden_channels'] = self.encode_model.hidden_channels_str
        log_dic['negative_sampling_ratio'] = self.negative_sampling_ratio
        log_dic['num_epochs'] = self.num_epochs
        log_dic['validation'] = validation
        if validation is True:
            log_dic['best_epoch'] = self.best_epoch
        else:
            log_dic['best_epoch'] = None
        log_dic['auc'] = auc
        log_dic['true_positive'] = c_matrix[1,1]
        log_dic['false_positive'] = c_matrix[0,1]
        log_dic['true_negative'] = c_matrix[0,0]
        log_dic['false_negative'] = c_matrix[1,0]
        log_dic['accuracy'] = (c_matrix[1,1]+c_matrix[0,0])/c_matrix.sum().sum()
        log_dic['precision'] = c_matrix[1,1]/(c_matrix[1,1]+c_matrix[0,1])
        log_dic['recall'] = c_matrix[1,1]/(c_matrix[1,1]+c_matrix[1,0])
        log_dic['path'] = f"./output/{self.dataset_name}/{self.encode_modelname}/{self.decode_modelname}/activation_{self.activation}/sigmoidbias_{'True' if self.sigmoid_bias is True else 'False'}/numlayers_{self.num_layers}/negative_sampling_ratio_{self.negative_sampling_ratio}/epochs_{self.num_epochs}/{self.start_time.strftime('%Y%m%d_%H%M')}"
        self.summary = self.summary.append(pd.Series(log_dic), ignore_index=True)
        self.summary.to_csv(path_csv, index=False)

        return None