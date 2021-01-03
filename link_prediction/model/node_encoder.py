"""
graph neural networks for link prediction

    link preiction モデルのclassを定義する

Todo:

"""

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, global_mean_pool, JumpingKnowledge

from link_prediction.my_util import my_utils

class NN(torch.nn.Module):       
    '''Fully Connected Layer

    Model which applies fully connected layers to node features

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        lins (torch.nn.ModuleList): list of fully connected layers.
        convs (torch.nn.ModuleList): list of torch_geometric.nn.GCNConv (which is not used in NN class.)
        batchnorms (torch.nn.ModuleList): list of batch normalization.
        activation (obj`int` or None): activation function. None, "relu", "leaky_relu", or "tanh". (Default: None)
        dropout (float): Dropout ratio.        
        num_layers (int or None): the number of hidden layers.     
        hidden_channels_str (str): the number of output channels of each layer.
    '''
    def __init__(self, data, decode_modelname, num_hidden_channels = None, num_layers = None, hidden_channels = None, activation = None, dropout = 0.0):
        '''
        Args:
            data (torch_geometric.data.Data): graph data.
            decode_modelname
            num_hidden_channels (int or None): the number of output channels. If set to int, the same number of output channels is applied to all the layers . (Default: None)
            num_layers (int or None): the number of hidden layers. (Default: None)
            hidden_channels (list of int, or None): list of the number of output channels of each layer. If set, num_hidden_channels and num_layers are invalud. (Default: None)
            activation (obj`int` or None): activation function. None, "relu", "leaky_relu", or "tanh". (Default: None)
            dropout (float): Dropout ratio.        
        '''
        super(NN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_modelname = decode_modelname

        self.lins = torch.nn.ModuleList()

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(1, 1))

        self.batchnorms = torch.nn.ModuleList()

        if hidden_channels is None:
            self.hidden_channels_str = (f'{num_hidden_channels}_'*num_layers)[:-1]

            self.num_layers = num_layers

            self.lins.append(torch.nn.Linear(data.x.size(1), num_hidden_channels))
            for _ in range(num_layers - 1):
                self.lins.append(torch.nn.Linear(num_hidden_channels, num_hidden_channels))

            for _ in range(num_layers - 1):
                self.batchnorms.append(torch.nn.BatchNorm1d(num_hidden_channels))

            if self.decode_modelname == 'VGAE':
                self.lins.append(torch.nn.Linear(num_hidden_channels, num_hidden_channels))
                
        else:
            self.hidden_channels_str = ''
            for num in hidden_channels:
                self.hidden_channels_str += str(num)+'_'
            self.hidden_channels_str = self.hidden_channels_str[:-1]

            self.num_layers = len(hidden_channels)

            self.lins.append(torch.nn.Linear(data.x.size(1), hidden_channels[0]))
            for i in range(len(hidden_channels) - 1):
                self.lins.append(torch.nn.Linear(hidden_channels[i], hidden_channels[i+1]))

            for i in range(len(hidden_channels) - 1):
                self.batchnorms.append(torch.nn.BatchNorm1d(hidden_channels[i]))

            if self.decode_modelname == 'VGAE':
                self.lins.append(torch.nn.Linear(hidden_channels[-2], hidden_channels[-1]))

        self.activation = activation
        self.dropout = dropout

    def forward(self, x, edge_inde = None):
        '''
        Parameters:
            x (torch.tensor[num_nodes, input_channels]): input of the model (node features).
            edge_index (torch.tensor[2, num_edges]): tonsor of the pair of node indexes with edges.

        Returns:
            z (torch.tensor[num_nodes, output_channels]): node-wise latent features.
        '''
        z = x
        if self.decode_modelname == 'VGAE':
            for i, lin in enumerate(self.lins[:-2]):
                z = F.dropout(z, self.dropout, training = self.training)
                z = lin(z)
                z = self.batchnorms[i](z)
                if self.activation == "relu":
                    z = z.relu()
                elif self.activation == "leaky_relu":
                    z = F.leaky_relu(z, negative_slope=0.01)
                elif self.activation == "tanh":
                    z = torch.tanh(z)

            return self.lins[-2](z), self.lins[-1](z)

        else:
            for i, lin in enumerate(self.lins):
                z = F.dropout(z, self.dropout, training = self.training)
                z = lin(z)
                if i < len(self.lins) - 1:
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)

            return z

class GCN(torch.nn.Module):  
    '''Vanilla Graph Neural Network

    ノードの特徴量に対してVanilla GCNを適用するモデル

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        lins (torch.nn.ModuleList): 全結合層のリスト.
        convs (torch.nn.ModuleList): torch_geometric.nn.GCNConvのリスト.
        batchnorms (torch.nn.ModuleList): batch normalizationのリスト.
        activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
        dropout (float): 各層のDropoutの割合. 
        num_layers (int): 隠れ層の数.
        hidden_channels_str (str): 各層の出力の次元を文字列として記録.
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
    '''     
    def __init__(self, data, decode_modelname, train_pos_edge_adj_t, num_hidden_channels = None, num_layers = None, hidden_channels = None, activation = None, dropout = 0.0):
        '''
        Args:
            data (torch_geometric.data.Data): グラフデータ.
            decode_modelname
            train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
            num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される. (Default: None)
            num_layers (int or None): 隠れ層の数. (Default: None)
            hidden_channels (list of int, or None): 各隠れ層の出力の配列. 指定するとnum_hidden_channels とnum_layersは無効化される. (Default: None)
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            dropout (float): 各層のDropoutの割合. (Default: 0.0)
        '''
        super(GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_modelname = decode_modelname

        self.train_pos_edge_adj_t = train_pos_edge_adj_t

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(1, 1))
        
        self.convs = torch.nn.ModuleList()

        self.batchnorms = torch.nn.ModuleList()

        if hidden_channels is None:
            self.hidden_channels_str = (f'{num_hidden_channels}_'*num_layers)[:-1]

            self.num_layers = num_layers
            self.convs.append(GCNConv(data.x.size(1), num_hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(num_hidden_channels, num_hidden_channels))

            for _ in range(num_layers - 1):
                self.batchnorms.append(torch.nn.BatchNorm1d(num_hidden_channels))

            if self.decode_modelname == 'VGAE':
                self.convs.append(GCNConv(num_hidden_channels, num_hidden_channels))

        else:            
            self.hidden_channels_str = ''
            for num in hidden_channels:
                self.hidden_channels_str += str(num)+'_'
            self.hidden_channels_str = self.hidden_channels_str[:-1]

            self.num_layers = len(hidden_channels)
            self.convs.append(GCNConv(data.x.size(1), hidden_channels[0]))
            for i in range(len(hidden_channels) - 1):
                self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))

            for i in range(len(hidden_channels) - 1):
                self.batchnorms.append(torch.nn.BatchNorm1d(hidden_channels[i]))

            if self.decode_modelname == 'VGAE':
                self.convs.append(GCNConv(hidden_channels[-2], hidden_channels[-1]))

        self.activation = activation
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        '''
        Parameters:
            x (torch.tensor[num_nodes, input_channels]): input of the model (node features).
            edge_index (torch.tensor[2, num_edges]): tonsor of the pair of node indexes with edges.

        Returns:
            z (torch.tensor[num_nodes, output_channels]): node-wise latent features.
        '''
        if edge_index is None:
            edge_index = self.train_pos_edge_adj_t

        z = x
        if self.decode_modelname == 'VGAE':
            for i, conv in enumerate(self.convs[:-2]):
                z = F.dropout(z, self.dropout, training = self.training)
                z = conv(z, edge_index)
                z = self.batchnorms[i](z)
                if self.activation == "relu":
                    z = z.relu()
                elif self.activation == "leaky_relu":
                    z = F.leaky_relu(z, negative_slope=0.01)
                elif self.activation == "tanh":
                    z = torch.tanh(z)

            return self.convs[-2](z, edge_index), self.convs[-1](z, edge_index)

        else:
            for i, conv in enumerate(self.convs):
                z = F.dropout(z, self.dropout, training = self.training)
                z = conv(z, edge_index)
                if i < len(self.convs) - 1:
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)

            return z

class GCNII(torch.nn.Module):       
    '''GCNII

    ノードの特徴量に対してGCNIIを適用するモデル

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        lins (torch.nn.ModuleList): 全結合層のリスト.
        convs (torch.nn.ModuleList): torch_geometric.nn.GCNConvのリスト.
        batchnorms (torch.nn.ModuleList): batch normalizationのリスト.
        activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
        dropout (float): 各層のDropoutの割合. 
        num_layers (int or None): 隠れ層の数.
        hidden_channels_str (str): 各層の出力の次元を文字列として記録
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
    '''    

    def __init__(self, data, decode_modelname, train_pos_edge_adj_t, num_hidden_channels, num_layers, alpha=0.1, theta=0.5, shared_weights = True, activation = None, dropout = 0.0):
        '''
        Args:
            data (torch_geometric.data.Data): グラフデータ.
            decode_modelname
            train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
            num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される.
            num_layers (int or None): 隠れ層の数.
            alpha (float): convolution後に初期層を加える割合. (Default: 0.1)
            theta (float): .
            shared_weights (bool): . (Default: True)
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            dropout (float): 各層のDropoutの割合. (Default: 0.0)
        '''
        super(GCNII, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_modelname = decode_modelname

        self.num_layers = num_layers
        self.train_pos_edge_adj_t = train_pos_edge_adj_t
        self.hidden_channels_str = (f'{num_hidden_channels}_'*num_layers)[:-1]

        self.lins = torch.nn.ModuleList()
        # self.lins.append(GCNConv(data.x.size(1), num_hidden_channels))
        self.lins.append(GATConv(in_channels=data.x.size(1), out_channels=num_hidden_channels, heads=1, concat=True, dropout=dropout))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers-1):
            self.convs.append(GCN2Conv(num_hidden_channels, alpha, theta, layer+1, shared_weights = shared_weights, normalize = False))

        if self.decode_modelname == 'VGAE':
            self.convs.append(GCN2Conv(num_hidden_channels, alpha, theta, layer+1, shared_weights = shared_weights, normalize = False))

        self.batchnorms = torch.nn.ModuleList()
        for layer in range(num_layers - 1):
            self.batchnorms.append(torch.nn.BatchNorm1d(num_hidden_channels))

        self.activation = activation
        self.dropout = dropout    

    def forward(self, x, edge_index=None):
        '''
        Parameters:
            x (torch.tensor[num_nodes, input_channels]): input of the model (node features).
            edge_index (torch.tensor[2, num_edges]): tonsor of the pair of node indexes with edges.

        Returns:
            z (torch.tensor[num_nodes, output_channels]): node-wise latent features.
        '''
        if edge_index is None:
            edge_index = self.train_pos_edge_adj_t

        # 線形変換で次元削減して入力とする
        z = self.lins[0](x, edge_index)

        x_0 = z

        if self.decode_modelname == 'VGAE':
            for i, conv in enumerate(self.convs[:-2]):
                z = F.dropout(z, self.dropout, training = self.training)
                z = conv(z, x_0, edge_index)
                z = self.batchnorms[i](z)
                if self.activation == "relu":
                    z = z.relu()
                elif self.activation == "leaky_relu":
                    z = F.leaky_relu(z, negative_slope=0.01)
                elif self.activation == "tanh":
                    z = torch.tanh(z)

            return self.convs[-2](z, x_0, edge_index), self.convs[-1](z, x_0, edge_index)

        else:
            for i, conv in enumerate(self.convs):
                z = F.dropout(z, self.dropout, training = self.training)
                if i == len(self.convs)-1:
                    z = conv(z, edge_index)
                else:
                    z = conv(z, x_0, edge_index)
                if i < len(self.convs) - 1:
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)

            return z


class GCNIIwithJK(torch.nn.Module):       
    '''GCNII with Jumping Knowledge

    ノードの特徴量に対してGCNIIを適用し、Jumping Knowledgeでマルチスケール化

    Attributes:
        device (:obj:`int`): 'cpu', 'cuda'. 
        lins (torch.nn.ModuleList): 全結合層のリスト.
        convs (torch.nn.ModuleList): torch_geometric.nn.GCNConvのリスト.
        jk_mode (:obj:`str`): JK-Netにおけるaggregation方法. ('cat', 'max' or 'lstm'). (Default: 'cat)
        batchnorms (torch.nn.ModuleList): batch normalizationのリスト.
        activation (:obj:`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
        dropout (float): 各層のDropoutの割合. 
        num_layers (int or None): 隠れ層の数.
        hidden_channels_str (str): 各層の出力の次元を文字列として記録
        train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
    '''    

    def __init__(self, data, decode_modelname, train_pos_edge_adj_t, num_hidden_channels, num_layers, jk_mode='cat', alpha=0.1, theta=0.5, shared_weights = True, activation = None, dropout = 0.0):
        '''
        Args:
            data (torch_geometric.data.Data): グラフデータ.
            decode_modelname
            train_pos_edge_adj_t (torch.SparseTensor[2, num_pos_edges]): trainデータのリンク.
            num_hidden_channels (int or None): 隠れ層の出力次元数. 全ての層で同じ値が適用される.
            num_layers (int or None): 隠れ層の数.
            jk_mode (:obj:`str`): JK-Netにおけるaggregation方法. ('cat', 'max' or 'lstm'). (Default: 'cat)
            alpha (float): convolution後に初期層を加える割合. (Default: 0.1)
            theta (float): .
            shared_weights (bool): . (Default: True)
            activation (obj`int` or None): activation functionを指定。None, "relu", "leaky_relu", or "tanh". (Default: None)
            dropout (float): 各層のDropoutの割合. (Default: 0.0)
        '''
        super(GCNIIwithJK, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decode_modelname = decode_modelname

        self.num_layers = num_layers
        self.train_pos_edge_adj_t = train_pos_edge_adj_t
        self.hidden_channels_str = (f'{num_hidden_channels}_'*num_layers)[:-1]

        self.lins = torch.nn.ModuleList()
        # self.lins.append(torch.nn.Linear(data.x.size(1), num_hidden_channels))
        self.lins.append(GCNConv(data.x.size(1), num_hidden_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(num_hidden_channels, alpha, theta, layer+1, shared_weights = shared_weights, normalize = False))

        if self.decode_modelname == 'VGAE':
            self.convs.append(GCN2Conv(num_hidden_channels, alpha, theta, layer+1, shared_weights = shared_weights, normalize = False))

        self.jk_mode = jk_mode
        self.jumps = torch.nn.ModuleList()
        for layer in range(num_layers//4):
            self.jumps.append(JumpingKnowledge(jk_mode))

        if self.jk_mode == 'cat':
            self.lins.append(torch.nn.Linear(4 * num_hidden_channels, num_hidden_channels))

        self.batchnorms = torch.nn.ModuleList()
        for layer in range(num_layers - 1 + (num_layers%4==0)):
            self.batchnorms.append(torch.nn.BatchNorm1d(num_hidden_channels))

        self.activation = activation
        self.dropout = dropout    

    def forward(self, x, edge_index=None):
        '''
        Parameters:
            x (torch.tensor[num_nodes, input_channels]): input of the model (node features).
            edge_index (torch.tensor[2, num_edges]): tonsor of the pair of node indexes with edges.

        Returns:
            z (torch.tensor[num_nodes, output_channels]): node-wise latent features.
        '''
        if edge_index is None:
            edge_index = self.train_pos_edge_adj_t

        # 線形変換で次元削減して入力とする
        # x = F.dropout(self.data.x, self.dropout, training=self.training)
        z = self.lins[0](x, edge_index)

        x_0 = z

        if self.decode_modelname == 'VGAE':
            zs = []
            for i, conv in enumerate(self.convs[:-2]):
                z = F.dropout(z, self.dropout, training = self.training)
                z = conv(z, x_0, edge_index)
                zs += [z]
                if (i < len(self.convs[:-2])) or (i%4==3):
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)

                if i%4 == 3:
                    z = self.jumps[i//4](zs[4*(i//4):4*((i//4)+1)])
                    # z = self.jumps[i//4](zs)
                    # zs = []
                    x_0 = z
                    if self.jk_mode == 'cat':
                        z = self.lins[-1](z)
                        x_0 = z
            return self.convs[-2](z, x_0, edge_index), self.convs[-1](z, x_0, edge_index)

        else:
            zs = []
            for i, conv in enumerate(self.convs):
                z = F.dropout(z, self.dropout, training = self.training)
                z = conv(z, x_0, edge_index)
                zs += [z]
                if (i < len(self.convs) - 1) or (i%4==3):
                    z = self.batchnorms[i](z)
                    if self.activation == "relu":
                        z = z.relu()
                    elif self.activation == "leaky_relu":
                        z = F.leaky_relu(z, negative_slope=0.01)
                    elif self.activation == "tanh":
                        z = torch.tanh(z)

                if i%4 == 3:
                    z = self.jumps[i//4](zs[4*(i//4):4*((i//4)+1)])
                    # z = self.jumps[i//4](zs)
                    # zs = []
                    x_0 = z
                    if self.jk_mode == 'cat':
                        z = self.lins[-1](z)
                        x_0 = z
            return z